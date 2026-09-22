import os
from copy import deepcopy
from math import exp
from pathlib import Path
from typing import List, Optional, Type, Union

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from omegaconf import ListConfig
from torch import Tensor, no_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, _LRScheduler

from yolo.config.config import (
    IDX_TO_ID,
    DataConfig,
    NMSConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from yolo.model.yolo import YOLO
from yolo.utils.bounding_box_utils import Anc2Box, Vec2Box, bbox_nms, transform_bbox
from yolo.utils.ema_utils import foreach_ema_update
from yolo.utils.logger import logger
from yolo.utils.nms_utils import select_nms_free_detections


def lerp(start: float, end: float, step: Union[int, float], total: int = 1):
    """
    Linearly interpolates between start and end values.

    start * (1 - step) + end * step

    Parameters:
        start (float): The starting value.
        end (float): The ending value.
        step (int): The current step in the interpolation process.
        total (int): The total number of steps.

    Returns:
        float: The interpolated value.
    """
    return start + (end - start) * step / total


class EMA(Callback):
    """Maintain EMA on successful optimizer updates, including under AMP."""

    def __init__(self, decay: float = 0.9999, tau: float = 2000):
        super().__init__()
        logger.info(":chart_with_upwards_trend: Enable Model EMA")
        self.decay = decay
        self.tau = tau
        self.step = 0
        self.ema_state_dict = None
        self._step_handles = []

    def setup(self, trainer, pl_module, stage):
        if hasattr(pl_module.model, "qat_metadata"):
            raise ValueError(
                "Disable EMA for QAT: averaging observer ranges/scales would corrupt the quantization state."
            )
        pl_module.ema = deepcopy(pl_module.model).eval()
        pl_module.ema.requires_grad_(False)

    def _initialize(self, pl_module):
        if self.ema_state_dict is None:
            self.ema_state_dict = deepcopy(pl_module.model.state_dict())
        else:
            # Callback checkpoint states can be loaded on CPU before strategy setup.
            current = pl_module.model.state_dict()
            self.ema_state_dict = {key: value.to(current[key]) for key, value in self.ema_state_dict.items()}

    def on_train_start(self, trainer, pl_module):
        self._initialize(pl_module)
        for optimizer in trainer.optimizers:
            # An optimizer subclass calling super().step() can trigger PyTorch's
            # hooks twice. Count only the completed outermost optimizer call.
            depth = [0]

            def before_step(optimizer, args, kwargs, depth=depth):
                depth[0] += 1

            def after_step(optimizer, args, kwargs, depth=depth):
                depth[0] -= 1
                if depth[0] == 0:
                    self._after_optimizer_step(optimizer, pl_module)

            self._step_handles.append(optimizer.register_step_pre_hook(before_step))
            self._step_handles.append(optimizer.register_step_post_hook(after_step))

    def _after_optimizer_step(self, optimizer, pl_module):
        # Fused optimizers may execute step() but skip internally on overflow.
        found_inf = getattr(optimizer, "found_inf", None)
        if found_inf is None or not bool(found_inf):
            self.update(pl_module)

    def on_validation_start(self, trainer, pl_module):
        self._initialize(pl_module)
        pl_module.ema.load_state_dict(self.ema_state_dict)
        pl_module.ema.eval()

    @no_grad()
    def update(self, pl_module):
        self._initialize(pl_module)
        self.step += 1
        decay_factor = self.decay * (1 - exp(-self.step / self.tau))
        source = pl_module.model.state_dict()
        # Match the reference: integer counters are not exponentially averaged.
        floating = {key: value for key, value in self.ema_state_dict.items() if value.is_floating_point()}
        foreach_ema_update({key: source[key] for key in floating}, floating, decay_factor)
        self.ema_state_dict.update(floating)

    def teardown(self, trainer, pl_module, stage):
        for handle in self._step_handles:
            handle.remove()
        self._step_handles.clear()

    def state_dict(self):
        return {"step": self.step, "ema_state_dict": self.ema_state_dict}

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.ema_state_dict = state_dict["ema_state_dict"]


class GradientAccumulation(Callback):
    """Compatibility marker; TrainModel owns its manual accumulation schedule.

    Lightning's automatic accumulation divides the loss and forces an epoch-tail
    update. Neither operation matches the reference detector training loop.
    """

    def __init__(self, data_cfg: DataConfig, scheduler_cfg: SchedulerConfig):
        super().__init__()

    def setup(self, trainer, pl_module, stage):
        if stage == "fit" and pl_module.automatic_optimization:
            raise ValueError("GradientAccumulation requires TrainModel manual optimization.")


def create_optimizer(model: YOLO, optim_cfg: OptimizerConfig) -> Optimizer:
    """Create an optimizer for the given model parameters based on the configuration.

    Returns:
        An instance of the optimizer configured according to the provided settings.
    """
    optimizer_class: Type[Optimizer] = getattr(torch.optim, optim_cfg.type)

    bias_params = [p for name, p in model.named_parameters() if "bias" in name]
    norm_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" in name]
    conv_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" not in name]

    model_parameters = [
        {"params": bias_params, "momentum": 0.937, "weight_decay": 0},
        {"params": conv_params, "momentum": 0.937},
        {"params": norm_params, "momentum": 0.937, "weight_decay": 0},
    ]

    def next_epoch(self, batch_num, epoch_idx):
        self.min_lr = self.max_lr
        self.max_lr = [param["lr"] for param in self.param_groups]
        # TODO: load momentum from config instead a fix number
        #       0.937: Start Momentum
        #       0.8  : Normal Momemtum
        #       3    : The warm up epoch num
        self.min_mom = lerp(0.8, 0.937, min(epoch_idx, 3), 3)
        self.max_mom = lerp(0.8, 0.937, min(epoch_idx + 1, 3), 3)
        self.batch_num = batch_num
        self.batch_idx = 0

    def next_batch(self):
        self.batch_idx += 1
        lr_dict = dict()
        for lr_idx, param_group in enumerate(self.param_groups):
            min_lr, max_lr = self.min_lr[lr_idx], self.max_lr[lr_idx]
            param_group["lr"] = lerp(min_lr, max_lr, self.batch_idx, self.batch_num)
            param_group["momentum"] = lerp(self.min_mom, self.max_mom, self.batch_idx, self.batch_num)
            lr_dict[f"LR/{lr_idx}"] = param_group["lr"]
            lr_dict[f"momentum/{lr_idx}"] = param_group["momentum"]
        return lr_dict

    optimizer_class.next_batch = next_batch
    optimizer_class.next_epoch = next_epoch

    optimizer = optimizer_class(model_parameters, **optim_cfg.args)
    optimizer.max_lr = [0.1, 0, 0]
    return optimizer


def create_scheduler(optimizer: Optimizer, schedule_cfg: SchedulerConfig) -> _LRScheduler:
    """Create a learning rate scheduler for the given optimizer based on the configuration.

    Returns:
        An instance of the scheduler configured according to the provided settings.
    """
    scheduler_class: Type[_LRScheduler] = getattr(torch.optim.lr_scheduler, schedule_cfg.type)
    schedule = scheduler_class(optimizer, **schedule_cfg.args)
    if hasattr(schedule_cfg, "warmup"):
        wepoch = schedule_cfg.warmup.epochs
        lambda1 = lambda epoch: (epoch + 1) / wepoch if epoch < wepoch else 1
        lambda2 = lambda epoch: 10 - 9 * ((epoch + 1) / wepoch) if epoch < wepoch else 1
        warmup_schedule = LambdaLR(optimizer, lr_lambda=[lambda2, lambda1, lambda1])
        schedule = SequentialLR(optimizer, schedulers=[warmup_schedule, schedule], milestones=[wepoch - 1])
    return schedule


def initialize_distributed() -> None:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f"🔢 Initialized process group; rank: {rank}, size: {world_size}")
    return local_rank


def get_device(device_spec: Union[str, int, List[int]]) -> torch.device:
    ddp_flag = False
    if isinstance(device_spec, (list, ListConfig)):
        ddp_flag = True
        device_spec = initialize_distributed()
    if torch.cuda.is_available() and "cuda" in str(device_spec):
        return torch.device(device_spec), ddp_flag
    if not torch.cuda.is_available():
        if device_spec != "cpu":
            logger.warning(f"❎ Device spec: {device_spec} not support, Choosing CPU instead")
        return torch.device("cpu"), False

    device = torch.device(device_spec)
    return device, ddp_flag


class PostProcess:
    """Decode and restore detections, selecting NMS or one-to-one top-k."""

    def __init__(self, converter: Union[Vec2Box, Anc2Box], nms_cfg: NMSConfig, *, nms_free: bool = False) -> None:
        self.converter = converter
        self.nms = nms_cfg
        self.nms_free = nms_free

    def __call__(
        self, predict, rev_tensor: Optional[Tensor] = None, image_size: Optional[List[int]] = None
    ) -> List[Tensor]:
        if image_size is not None:
            self.converter.update(image_size)
        prediction = self.converter(predict["Main"])
        pred_class, _, pred_bbox = prediction[:3]
        pred_conf = prediction[3] if len(prediction) == 4 else None
        if rev_tensor is not None:
            pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]
        if self.nms_free:
            scores = pred_class.sigmoid() * (1 if pred_conf is None else pred_conf)
            return select_nms_free_detections(pred_bbox, scores, self.nms.min_confidence, self.nms.max_bbox)
        return bbox_nms(pred_class, pred_bbox, self.nms, pred_conf)


def collect_prediction(predict_json: List, local_rank: int) -> List:
    """
    Collects predictions from all distributed processes and gathers them on the main process (rank 0).

    Args:
        predict_json (List): The prediction data (can be of any type) generated by the current process.
        local_rank (int): The rank of the current process. Typically, rank 0 is the main process.

    Returns:
        List: The combined list of predictions from all processes if on rank 0, otherwise predict_json.
    """
    if dist.is_initialized() and local_rank == 0:
        all_predictions = [None for _ in range(dist.get_world_size())]
        dist.gather_object(predict_json, all_predictions, dst=0)
        predict_json = [item for sublist in all_predictions for item in sublist]
    elif dist.is_initialized():
        dist.gather_object(predict_json, None, dst=0)
    return predict_json


def predicts_to_json(img_paths, predicts, rev_tensor):
    """
    TODO: function document
    turn a batch of imagepath and predicts(n x 6 for each image) to a List of diction(Detection output)
    """
    batch_json = []
    for img_path, bboxes, box_reverse in zip(img_paths, predicts, rev_tensor):
        scale, shift = box_reverse.split([1, 4])
        bboxes = bboxes.clone()
        bboxes[:, 1:5] = (bboxes[:, 1:5] - shift[None]) / scale[None]
        bboxes[:, 1:5] = transform_bbox(bboxes[:, 1:5], "xyxy -> xywh")
        for cls, *pos, conf in bboxes:
            bbox = {
                "image_id": int(Path(img_path).stem),
                "category_id": IDX_TO_ID[int(cls)],
                "bbox": [float(p) for p in pos],
                "score": float(conf),
            }
            batch_json.append(bbox)
    return batch_json
