"""Export detection models with one pre-NMS output and rank <= 4 tensors."""

import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from yolo.config.config import Config
from yolo.model.module import Anchor2Vec, MultiheadDetection
from yolo.model.yolo import create_model
from yolo.utils.bounding_box_utils import (
    Anc2Box,
    _strides_from_feature_maps,
    create_converter,
)
from yolo.utils.logger import logger


class ExportAnchor2Vec(nn.Module):
    """DFL expectation with rank-4 intermediates and the checkpoint's projection."""

    def __init__(self, source: Anchor2Vec):
        super().__init__()
        self.reg_max = source.anc2vec.in_channels
        self.projection = nn.Conv2d(self.reg_max, 1, 1, bias=False)
        self.projection.weight = nn.Parameter(
            source.anc2vec.weight.detach().clone().reshape(1, self.reg_max, 1, 1), requires_grad=False
        )

    def forward(self, anchor_x):
        batch, _, height, width = anchor_x.shape
        logits = anchor_x.reshape(batch, 4, self.reg_max, height * width).transpose(1, 2)
        vector = self.projection(logits.softmax(dim=1)).reshape(batch, 4, height, width)
        # Export only consumes vector; keep the unused logits rank-4 as well.
        return logits, vector


class ExportAnchorProbabilities(nn.Module):
    """DFL probabilities, grouped by L/T/R/B, without rank-5 tensors or projection."""

    def __init__(self, source: Anchor2Vec):
        super().__init__()
        self.reg_max = source.anc2vec.in_channels

    def forward(self, anchor_x):
        batch, _, height, width = anchor_x.shape
        logits = anchor_x.reshape(batch, 4, self.reg_max, height * width)
        # [B, H*W, 4, R]: normalize each direction over its own bins.
        probabilities = logits.permute(0, 3, 1, 2).softmax(dim=-1)
        return logits, probabilities.flatten(2)


class ExportModel(nn.Module):
    """NCHW RGB input -> one tensor containing Main detections.

    With probabilities=True, DFL heads return [B, N, 4*reg_max + classes]:
    L/T/R/B softmax distributions followed by sigmoid class scores. Otherwise
    return decoded [B, N, 4 + classes], pixel xyxy followed by class scores.
    Anchor-based YOLOv7 heads always use the decoded contract.

    Only Main detections are exported. No confidence filtering, top-k, NMS,
    clipping or inverse letterbox transform is performed.
    """

    def __init__(self, model, anchor_cfg, image_size, model_name, *, probabilities=False):
        super().__init__()
        self.model = deepcopy(model).cpu().float().eval()
        main = next((layer for layer in self.model.model if layer.tags == "Main" and layer.output), None)
        if type(main) is not MultiheadDetection:
            raise ValueError("Export requires a detection model with a Main MultiheadDetection output.")
        self.probabilities = probabilities and all(
            isinstance(getattr(head, "anc2vec", None), Anchor2Vec) for head in main.heads
        )
        self.class_num = model.num_classes
        for module in list(self.model.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, Anchor2Vec):
                    replacement = ExportAnchorProbabilities(child) if self.probabilities else ExportAnchor2Vec(child)
                    setattr(module, name, replacement)
        if self.probabilities:
            return
        with torch.no_grad():
            converter = create_converter(model_name, self.model, anchor_cfg, image_size, torch.device("cpu"))
        self.anchor_based = isinstance(converter, Anc2Box)
        if self.anchor_based:
            self.anchor_num = converter.anchor_num
            self.strides = tuple(converter.strides)
            for index, grid in enumerate(converter.anchor_grids):
                self.register_buffer(f"grid_{index}", grid.reshape(1, 1, -1, 2))
                self.register_buffer(f"scale_{index}", converter.anchor_scale[index].reshape(1, self.anchor_num, 1, 2))
        else:
            self.register_buffer("anchor_grid", converter.anchor_grid)
            self.register_buffer("scaler", converter.scaler.view(1, -1, 1))

    def forward(self, images):
        predictions = self.model(images, shortcut="Main")["Main"]
        if self.probabilities:
            return torch.cat(
                [torch.cat((head[2], head[0].flatten(2).transpose(1, 2).sigmoid()), dim=-1) for head in predictions],
                dim=1,
            )
        if self.anchor_based:
            outputs = []
            for index, prediction in enumerate(predictions):
                batch, _, height, width = prediction.shape
                # Keep decoding rank <= 4 for LiteRT broadcast legalization.
                prediction = prediction.reshape(batch, self.anchor_num, 5 + self.class_num, height * width)
                prediction = prediction.permute(0, 1, 3, 2).sigmoid()
                center = (prediction[..., :2] * 2 - 0.5 + getattr(self, f"grid_{index}")) * self.strides[index]
                size = (prediction[..., 2:4] * 2).square() * getattr(self, f"scale_{index}")
                boxes = torch.cat((center - size / 2, center + size / 2), dim=-1)
                scores = prediction[..., 5:] * prediction[..., 4:5]
                outputs.append(torch.cat((boxes, scores), dim=-1).reshape(batch, -1, 4 + self.class_num))
            return torch.cat(outputs, dim=1)

        scores = torch.cat([head[0].flatten(2).transpose(1, 2) for head in predictions], dim=1).sigmoid()
        distances = torch.cat([head[2].flatten(2).transpose(1, 2) for head in predictions], dim=1) * self.scaler
        left_top, right_bottom = distances.chunk(2, dim=-1)
        boxes = torch.cat((self.anchor_grid - left_top, self.anchor_grid + right_bottom), dim=-1)
        return torch.cat((boxes, scores), dim=-1)


def validate_onnx_tensor_ranks(model):
    """Infer all tensor ranks and reject unknown or >4 ranks, including constants."""
    import onnx

    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

    def check_tensor(tensor):
        if len(tensor.dims) > 4:
            raise ValueError(f"ONNX tensor {tensor.name!r} has rank {len(tensor.dims)}; maximum is 4.")

    def check_graph(graph):
        values = {v.name: v for v in [*graph.input, *graph.value_info, *graph.output]}
        for value in values.values():
            tensor_type = value.type.tensor_type
            if not tensor_type.HasField("shape"):
                raise ValueError(f"Cannot verify ONNX tensor rank: {value.name!r}.")
            if len(tensor_type.shape.dim) > 4:
                raise ValueError(f"ONNX tensor {value.name!r} has rank {len(tensor_type.shape.dim)}; maximum is 4.")
        for tensor in graph.initializer:
            check_tensor(tensor)
        for node in graph.node:
            for name in node.output:
                if name and name not in values:
                    raise ValueError(f"Cannot verify ONNX tensor rank: {name!r}.")
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    check_tensor(attribute.t)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for tensor in attribute.tensors:
                        check_tensor(tensor)
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    check_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        check_graph(subgraph)

    check_graph(model.graph)
    return model


def export_model(cfg: Config) -> Path:
    """Run an export without a Trainer, dataset download or experiment logger."""
    task = cfg.task
    qdq = getattr(task, "qdq", False)
    if qdq and task.format != "onnx":
        raise ValueError("QDQ export is ONNX-only; set task.format=onnx.")
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
    is_qat = hasattr(model, "qat_metadata")
    if is_qat != qdq:
        raise ValueError("Use task.qdq=true with a QAT checkpoint; FP export requires FP weights.")
    if qdq:
        from yolo.tools.qat import freeze_for_export

        freeze_for_export(model)
    wrapper = ExportModel(
        model, cfg.model.anchor, list(cfg.image_size), cfg.model.name, probabilities=task.format == "onnx"
    ).eval()
    if qdq:
        from yolo.tools.qat import snap_export_weights

        snap_export_weights(wrapper)
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
        exported = validate_onnx_tensor_ranks(onnx.load(str(output)))
        onnx.checker.check_model(exported)
        # Carry the decoder contract with the artifact, including custom heads.
        metadata = {
            "version": 1,
            "output_format": "dfl" if wrapper.probabilities else "xyxy",
            "class_num": wrapper.class_num,
            "class_names": list(cfg.dataset.class_list) if len(cfg.dataset.class_list) == wrapper.class_num else [],
        }
        if wrapper.probabilities:
            with torch.no_grad():
                heads = wrapper.model(sample, shortcut="Main")["Main"]
            metadata["strides"] = _strides_from_feature_maps((head[0] for head in heads), cfg.image_size)
            metadata["reg_max"] = (output_shape[-1] - wrapper.class_num) // 4
        properties = {"yolo.inference": json.dumps(metadata)}
        if qdq:
            from yolo.tools.qat import encoding_manifest

            encodings = encoding_manifest(wrapper)
            q_nodes = [node for node in exported.graph.node if node.op_type == "QuantizeLinear"]
            dq_nodes = [node for node in exported.graph.node if node.op_type == "DequantizeLinear"]
            if len(q_nodes) != len(encodings) or len(dq_nodes) != len(encodings):
                raise ValueError("Export did not preserve every QAT quantizer as a Q/DQ pair.")
            properties.update(
                {
                    "yolo.qat": json.dumps(model.qat_metadata),
                    "yolo.qat.encodings": json.dumps(encodings),
                    "yolo.output": "[B,N,4*reg_max+C]: left,top,right,bottom bins,class probabilities",
                },
            )
        onnx.helper.set_model_props(exported, properties)
        onnx.save(exported, str(output))
    else:
        edge_model = litert_torch.convert(wrapper, (sample,))
        edge_model.export(str(output))
    contract = "L/T/R/B softmax + class sigmoid" if wrapper.probabilities else "xyxy + class scores"
    logger.info(f"Exported {output}: input {tuple(sample.shape)}, output {output_shape} ({contract}, no NMS)")
    return output
