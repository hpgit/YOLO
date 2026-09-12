from math import isfinite
from pathlib import Path

from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.coco_eval import CocoJsonEvaluator
from yolo.utils.logger import logger
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler


def create_validation_metric(validation_cfg, dataset_cfg):
    """Prefer authoritative annotation JSON; retain tensor metrics for TXT datasets."""
    backend = getattr(validation_cfg, "evaluator", "auto")
    if backend not in {"auto", "coco", "torchmetrics"}:
        raise ValueError(f"Unknown validation evaluator: {backend!r}")
    dataset_root = Path(dataset_cfg.path)
    phase = dataset_cfg.get(validation_cfg.task, validation_cfg.task)
    configured_path = getattr(validation_cfg, "annotation_path", None)
    annotation_path = Path(configured_path) if configured_path else Path("annotations") / f"instances_{phase}.json"
    if not annotation_path.is_absolute():
        annotation_path = dataset_root / annotation_path
    # The loader gives an explicit split TXT list precedence over JSON labels.
    # Match that in auto mode; an explicit JSON request makes JSON authoritative.
    split_txt = (dataset_root / f"{phase}.txt").is_file()
    use_json = backend == "coco" or (
        backend == "auto" and (configured_path or (annotation_path.is_file() and not split_txt))
    )
    if use_json:
        if validation_cfg.data.data_augment:
            raise ValueError("COCO JSON evaluation requires data_augment={} (PadAndResize only).")
        metric = CocoJsonEvaluator(annotation_path, image_root=dataset_root / "images" / phase)
        if len(metric.coco_gt.getCatIds()) != dataset_cfg.class_num:
            raise ValueError("Annotation category count must match dataset.class_num for COCO evaluation.")
        logger.info(f"COCO JSON evaluation: {annotation_path}")
        return metric
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
    metric.warn_on_many_detections = False
    logger.info("Tensor GT evaluation (TorchMetrics); official COCO JSON is not used.")
    return metric


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)

    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.metric = create_validation_metric(self.validation_cfg, self.cfg.dataset)
        self.ema = self.model

    def setup(self, stage):
        # COCO evaluation reads scores one scalar at a time. Keeping states on
        # CPU avoids thousands of tiny CUDA copies. NCCL/DDP still needs GPU
        # states for TorchMetrics synchronization, so retain that path there.
        if self._trainer is not None and not isinstance(self.metric, CocoJsonEvaluator):
            self.metric.compute_on_cpu = self.trainer.world_size == 1
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images, shortcut="Main"), image_size=[W, H])
        if isinstance(self.metric, CocoJsonEvaluator):
            self.metric.update(predicts, img_paths, image_size=[W, H])
        else:
            self.metric.update(
                [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
            )
        # Batch AP is expensive and not the dataset AP; compute once at epoch end.
        return predicts, None

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        epoch_metrics.pop("classes", None)
        # COCO evaluator already gathers/deduplicates images and broadcasts one
        # global result. Averaging rank-local AP is not a valid COCO evaluation.
        sync_dist = not isinstance(self.metric, CocoJsonEvaluator)
        epoch_metrics = {key: value.to(self.device) for key, value in epoch_metrics.items()}
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=sync_dist)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=sync_dist,
        )
        self.metric.reset()


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.automatic_optimization = False
        self._last_opt_step = -1
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        batches = self.trainer.num_training_batches
        if not isfinite(batches) or batches <= 0:
            raise ValueError("Training requires a finite, nonempty number of batches.")
        self._batches_per_epoch = int(batches)
        if not hasattr(self, "_last_opt_step"):
            self._last_opt_step = -1
        self.trainer.optimizers[0].next_epoch(self._batches_per_epoch, self.current_epoch)
        # The reference discards an incomplete accumulation at the epoch boundary,
        # while preserving the absolute iteration of the preceding optimizer step.
        self.optimizers().zero_grad(set_to_none=True)
        self.vec2box.update(self.cfg.image_size)

    def accumulation_at(self, iteration):
        data = self.cfg.task.data
        global_batch = data.batch_size * self.trainer.world_size
        nominal_batch = getattr(data, "equivalent_batch_size", global_batch)
        ratio = nominal_batch / global_batch
        warmup = self.cfg.task.scheduler.warmup
        warmup_batches = max(round(warmup.epochs * self._batches_per_epoch),
                             getattr(warmup, "min_iterations", 100))
        if warmup_batches and iteration <= warmup_batches:
            return max(1, round(1 + (ratio - 1) * iteration / warmup_batches))
        return max(1, round(ratio))

    def on_before_optimizer_step(self, optimizer):
        # Lightning invokes this hook after GradScaler unscales gradients.
        self.clip_gradients(
            optimizer,
            gradient_clip_val=getattr(self.cfg.task, "gradient_clip_val", 10.0),
            gradient_clip_algorithm=getattr(self.cfg.task, "gradient_clip_algorithm", "norm"),
        )

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["training_accumulation"] = {"last_opt_step": self._last_opt_step}

    def on_load_checkpoint(self, checkpoint):
        self._last_opt_step = checkpoint.get("training_accumulation", {}).get("last_opt_step", -1)

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        batch_size, images, targets, *_ = batch
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"])
        main_predicts = self.vec2box(predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        batch_loss = loss * batch_size
        # DDP averages gradients across ranks; restore the reference global sum.
        self.manual_backward(batch_loss * self.trainer.world_size)
        iteration = self.current_epoch * self._batches_per_epoch + batch_idx
        if iteration - self._last_opt_step >= self.accumulation_at(iteration):
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self._last_opt_step = iteration
        return batch_loss.detach()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
