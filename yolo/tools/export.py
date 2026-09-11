"""Export detection models with one decoded, pre-NMS output tensor."""

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from yolo.config.config import Config
from yolo.model.module import MultiheadDetection
from yolo.model.yolo import create_model
from yolo.utils.bounding_box_utils import Anc2Box, create_converter
from yolo.utils.logger import logger


class ExportModel(nn.Module):
    """NCHW RGB input -> [B, N, 4 + classes], pixel xyxy and class scores.

    Only Main detections are exported. No confidence filtering, top-k, NMS,
    clipping or inverse letterbox transform is performed.
    """

    def __init__(self, model, anchor_cfg, image_size, model_name):
        super().__init__()
        self.model = model.cpu().float().eval()
        main = next((layer for layer in model.model if layer.tags == "Main" and layer.output), None)
        if type(main) is not MultiheadDetection:
            raise ValueError("Export requires a detection model with a Main MultiheadDetection output.")
        with torch.no_grad():
            converter = create_converter(model_name, model, anchor_cfg, image_size, torch.device("cpu"))
        self.anchor_based = isinstance(converter, Anc2Box)
        self.class_num = model.num_classes
        if self.anchor_based:
            self.anchor_num = converter.anchor_num
            self.strides = tuple(converter.strides)
            self.register_buffer("anchor_scale", converter.anchor_scale.reshape(-1, 1, self.anchor_num, 1, 2))
            for index, grid in enumerate(converter.anchor_grids):
                self.register_buffer(f"grid_{index}", grid.reshape(1, 1, -1, 2))
        else:
            self.register_buffer("anchor_grid", converter.anchor_grid)
            self.register_buffer("scaler", converter.scaler.view(1, -1, 1))

    def forward(self, images):
        predictions = self.model(images, shortcut="Main")["Main"]
        if self.anchor_based:
            outputs = []
            for index, prediction in enumerate(predictions):
                batch, _, height, width = prediction.shape
                # Keep decoding rank <= 4 for LiteRT broadcast legalization.
                prediction = prediction.reshape(batch, self.anchor_num, 5 + self.class_num, height * width)
                prediction = prediction.permute(0, 1, 3, 2).sigmoid()
                center = (prediction[..., :2] * 2 - 0.5 + getattr(self, f"grid_{index}")) * self.strides[index]
                size = (prediction[..., 2:4] * 2).square() * self.anchor_scale[index]
                boxes = torch.cat((center - size / 2, center + size / 2), dim=-1)
                scores = prediction[..., 5:] * prediction[..., 4:5]
                outputs.append(torch.cat((boxes, scores), dim=-1).reshape(batch, -1, 4 + self.class_num))
            return torch.cat(outputs, dim=1)

        scores = torch.cat([head[0].flatten(2).transpose(1, 2) for head in predictions], dim=1).sigmoid()
        distances = torch.cat([head[2].flatten(2).transpose(1, 2) for head in predictions], dim=1) * self.scaler
        left_top, right_bottom = distances.chunk(2, dim=-1)
        boxes = torch.cat((self.anchor_grid - left_top, self.anchor_grid + right_bottom), dim=-1)
        return torch.cat((boxes, scores), dim=-1)


def export_model(cfg: Config) -> Path:
    """Run an export without a Trainer, dataset download or experiment logger."""
    task = cfg.task
    if task.format not in ("onnx", "tflite"):
        raise ValueError("task.format must be 'onnx' or 'tflite'.")
    if not isinstance(task.batch_size, int) or isinstance(task.batch_size, bool) or task.batch_size < 1:
        raise ValueError("task.batch_size must be a positive integer.")
    if len(cfg.image_size) != 2 or any(not isinstance(n, int) or n <= 0 or n % 32 for n in cfg.image_size):
        raise ValueError("image_size must be [width, height], both positive multiples of 32.")
    if task.format == "tflite" and task.dynamic_batch:
        raise ValueError("TFLite export uses a fixed batch size; set task.dynamic_batch=false.")
    if task.format == "onnx" and task.opset < 13:
        raise ValueError("ONNX export requires task.opset >= 13.")

    # Fail before model construction/download when an optional backend is missing.
    if task.format == "onnx":
        try:
            import onnx
        except ImportError as exc:
            raise ImportError('ONNX export requires: pip install -e ".[export-onnx]"') from exc
    else:
        try:
            import litert_torch
        except ImportError as exc:
            raise ImportError('TFLite export requires: pip install -e ".[export-tflite]" (Python >= 3.11)') from exc

    output = (
        Path(task.output)
        if task.output
        else Path(cfg.out_path) / "export" / cfg.name / f"{cfg.model.name}.{task.format}"
    )
    if output.suffix != f".{task.format}":
        raise ValueError(f"task.output must end in .{task.format}")
    if output.exists() and not cfg.exist_ok:
        raise FileExistsError(output)
    model = create_model(deepcopy(cfg.model), class_num=cfg.dataset.class_num, weight_path=cfg.weight)
    wrapper = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name).eval()
    width, height = cfg.image_size
    sample = torch.zeros(task.batch_size, 3, height, width)
    with torch.no_grad():
        output_shape = tuple(wrapper(sample).shape)
    output.parent.mkdir(parents=True, exist_ok=True)
    if task.format == "onnx":
        dynamic_axes = {"images": {0: "batch_size"}, "predictions": {0: "batch_size"}} if task.dynamic_batch else None
        torch.onnx.export(
            wrapper,
            sample,
            str(output),
            input_names=["images"],
            output_names=["predictions"],
            opset_version=task.opset,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        onnx.checker.check_model(str(output))
    else:
        edge_model = litert_torch.convert(wrapper, (sample,))
        edge_model.export(str(output))
    logger.info(f"Exported {output}: input {tuple(sample.shape)}, output {output_shape} (xyxy + class scores, no NMS)")
    return output
