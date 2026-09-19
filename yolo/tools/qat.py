"""Deploy-structure convolution QAT with portable ONNX Q/DQ encodings.

This is a mixed graph: W8A8 Conv2d boundaries, float bias/nonlinearities/merges
and float probability outputs. Vendor compilation is a separate deployment step.
"""

from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.nn.utils.fusion import fuse_conv_bn_eval

from yolo.model.module import Anchor2Vec, Conv, MultiheadDetection, RepConv


class TrainingFakeQuantize(FakeQuantize):
    """Collect ranges only in train mode; validation must never calibrate."""

    def forward(self, x):
        if self.training:
            return super().forward(x)
        # Initial stride probes/sanity validation are FP and must not calibrate.
        # The export entry point rejects this state before tracing.
        if not torch.onnx.is_in_onnx_export():
            minimum = self.activation_post_process.min_val
            if minimum.numel() == 0 or not torch.isfinite(minimum).all():
                return x
        if self.fake_quant_enabled[0] == 0:
            return x
        if self.is_per_channel:
            return torch.fake_quantize_per_channel_affine(
                x, self.scale, self.zero_point, self.ch_axis, self.quant_min, self.quant_max
            )
        return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zero_point, self.quant_min, self.quant_max)


def activation_quantizer(averaging_constant):
    return TrainingFakeQuantize(
        observer=MovingAverageMinMaxObserver,
        averaging_constant=averaging_constant,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
    )


class QATConv2d(nn.Conv2d):
    """Use STE fake quantization on input, per-output-channel weight and output."""

    @classmethod
    def from_float(cls, source, averaging_constant):
        module = cls(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
        ).to(source.weight)
        module.weight = source.weight
        module.bias = source.bias
        module.input_fake_quant = activation_quantizer(averaging_constant)
        module.weight_fake_quant = TrainingFakeQuantize(
            observer=MovingAveragePerChannelMinMaxObserver,
            averaging_constant=averaging_constant,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            quant_min=-128,
            quant_max=127,
        )
        module.output_fake_quant = activation_quantizer(averaging_constant)
        return module.to(source.weight.device)

    def forward(self, x):
        x = self.input_fake_quant(x)
        weight = self.weight_fake_quant(self.weight)
        return self.output_fake_quant(self._conv_forward(x, weight, self.bias))


def quantizers(model):
    return ((name, module) for name, module in model.named_modules() if isinstance(module, TrainingFakeQuantize))


def deploy_model(model):
    """In-place Main-only conversion, before any observers or optimizer exist."""
    model.eval()
    main_index = next((i for i, layer in enumerate(model.model) if layer.tags == "Main" and layer.output), None)
    if main_index is None or type(model.model[main_index]) is not MultiheadDetection:
        raise ValueError("QAT requires a YOLOv9 Main detection head.")
    if not all(isinstance(getattr(head, "anc2vec", None), Anchor2Vec) for head in model.model[main_index].heads):
        raise ValueError("QAT currently supports YOLOv9 DFL heads only.")
    model.model = model.model[: main_index + 1]
    model.layer_index = {key: index for key, index in model.layer_index.items() if index <= main_index + 1}
    # Structural fusion must precede Conv-BN folding of the remaining blocks.
    for module in list(model.modules()):
        if isinstance(module, RepConv) and not hasattr(module, "reparam"):
            large = fuse_conv_bn_eval(module.conv1.conv, module.conv1.bn)
            small = fuse_conv_bn_eval(module.conv2.conv, module.conv2.bn)
            if large.stride != small.stride or large.groups != small.groups or small.kernel_size != (1, 1):
                raise ValueError("Unsupported RepConv geometry for deploy fusion.")
            pad_h, pad_w = large.kernel_size[0] // 2, large.kernel_size[1] // 2
            if large.padding != (pad_h * large.dilation[0], pad_w * large.dilation[1]) or small.padding != (0, 0):
                raise ValueError("RepConv fusion requires centered, same-padding branches.")
            with torch.no_grad():
                large.weight.add_(torch.nn.functional.pad(small.weight, (pad_w, pad_w, pad_h, pad_h)))
                large.bias.add_(small.bias)
            # Fused BN shifts become biases; keep them trainable during QAT.
            large.requires_grad_(any(parameter.requires_grad for parameter in module.parameters()))
            module.reparam = large
            del module.conv1, module.conv2
    for module in model.modules():
        if isinstance(module, Conv) and isinstance(module.bn, nn.BatchNorm2d):
            trainable = any(parameter.requires_grad for parameter in module.parameters())
            module.conv = fuse_conv_bn_eval(module.conv, module.bn).requires_grad_(trainable)
            module.bn = nn.Identity()
    return model


def prepare_qat(model, config):
    options = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    if hasattr(model, "qat_metadata"):
        if model.qat_metadata["config"] != options:
            raise ValueError("QAT configuration differs from checkpoint.")
        return model
    training = model.training
    start, freeze = options["fake_quant_start_epoch"], options["observer_freeze_epoch"]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (start, freeze)) or freeze <= start:
        raise ValueError("QAT epochs must satisfy 0 <= fake_quant_start_epoch < observer_freeze_epoch.")
    if not 0 < options["averaging_constant"] <= 1:
        raise ValueError("qat.averaging_constant must be in (0, 1].")
    deploy_model(model)
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.Conv2d:
                setattr(parent, name, QATConv2d.from_float(child, options["averaging_constant"]))
    model.qat_metadata = {
        "version": 1,
        "profile": "conv_w8a8",
        "config": options,
        "model_name": model.model_name,
        "class_num": model.num_classes,
        "reg_max": model.reg_max,
    }
    set_qat_epoch(model, 0)
    return model.train(training)


def set_qat_epoch(model, epoch):
    if not hasattr(model, "qat_metadata"):
        return
    options = model.qat_metadata["config"]
    for _, module in quantizers(model):
        module.enable_fake_quant(epoch >= options["fake_quant_start_epoch"])
        module.enable_observer(epoch < options["observer_freeze_epoch"])


def load_qat_state(model, checkpoint):
    metadata = checkpoint["qat"]
    if metadata["version"] != 1 or metadata["profile"] != "conv_w8a8":
        raise ValueError("Unsupported QAT checkpoint version/profile.")
    for key, actual in [("model_name", model.model_name), ("class_num", model.num_classes), ("reg_max", model.reg_max)]:
        if metadata[key] != actual:
            raise ValueError(f"QAT checkpoint {key} mismatch: {metadata[key]} != {actual}")
    prepare_qat(model, metadata["config"])
    state = checkpoint.get("model_state_dict")
    if state is None:
        state = {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
    model.load_state_dict(state, strict=True)


def checkpoint_weights(model):
    if hasattr(model, "qat_metadata"):
        return {"qat": deepcopy(model.qat_metadata), "model_state_dict": model.state_dict()}
    return model.model.state_dict()


def freeze_for_export(model):
    """Fail closed for uncalibrated/warmup checkpoints; never calibrate on dummy input."""
    if not hasattr(model, "qat_metadata"):
        raise ValueError("QDQ export requires a trained QAT checkpoint, not FP weights.")
    for name, module in quantizers(model):
        observer = module.activation_post_process
        if (
            observer.min_val.numel() == 0
            or not torch.isfinite(observer.min_val).all()
            or not torch.isfinite(observer.max_val).all()
        ):
            raise ValueError(f"Uninitialized QAT observer: {name}. Run training on representative data first.")
        if not module.fake_quant_enabled[0]:
            raise ValueError(f"QAT fake quantization is disabled at {name}; checkpoint is still in warmup.")
        if not torch.isfinite(module.scale).all() or not (module.scale > 0).all():
            raise ValueError(f"Invalid QAT scale: {name}")
        module.disable_observer()
    model.eval()


def encoding_manifest(model):
    return {
        name: {
            "scale": module.scale.detach().cpu().tolist(),
            "zero_point": module.zero_point.detach().cpu().tolist(),
            "dtype": "int8" if module.is_per_channel else "uint8",
            "axis": 0 if module.is_per_channel else None,
        }
        for name, module in quantizers(model)
    }


@torch.no_grad()
def snap_export_weights(model):
    """Materialize the trained weight grid in the export copy before Q/DQ.

    PyTorch uses reciprocal multiplication while ONNX specifies division.
    At half-bin ties, exporting the original float weight can select a different
    integer. Grid-aligned weights make Q idempotent and preserve trained rounding.
    The scales and zero-points, and the original training weights, stay unchanged.
    """
    for module in model.modules():
        if isinstance(module, QATConv2d):
            module.weight.copy_(module.weight_fake_quant(module.weight))


def configure_qat_run(cfg, checkpoint_path=None):
    """Restore the profile before constructing the Trainer, optimizer or model."""
    if not hasattr(cfg, "qat"):
        return checkpoint_path
    candidate = checkpoint_path or (Path(cfg.weight) if isinstance(cfg.weight, (str, Path)) else None)
    metadata = None
    if candidate and candidate.is_file():
        metadata = torch.load(candidate, map_location="cpu", weights_only=False).get("qat")
    if metadata:
        if cfg.qat.enabled and OmegaConf.to_container(cfg.qat, resolve=True) != metadata["config"]:
            raise ValueError("QAT configuration differs from checkpoint; resume with the saved QAT settings.")
        cfg.qat = OmegaConf.create(metadata["config"])
    elif checkpoint_path is not None and cfg.qat.enabled:
        # An explicit FP snapshot starts fresh QAT optimization from its weights.
        if isinstance(cfg.weight, (str, Path)):
            checkpoint_path = None
        else:
            raise ValueError("The named run contains FP training. Choose a new name for QAT fine-tuning.")
    return checkpoint_path
