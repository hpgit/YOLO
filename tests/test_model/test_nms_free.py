"""Dual-head routing and checkpoint compatibility for NMS-free YOLOv9."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from yolo.model.yolo import YOLO, create_model


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(nms_free=True, version=None):
    head_args = {"version": version} if version is not None else {}
    cfg = OmegaConf.create(
        {
            "name": "v9-test",
            "nms_free": nms_free,
            "anchor": {"reg_max": 4},
            "model": {
                "backbone": [{"Conv": {"source": 0, "args": {"out_channels": 16, "kernel_size": 3}}}],
                "detection": [
                    {"MultiheadDetection": {"source": [1], "tags": "Main", "output": True, "args": head_args}}
                ],
                "auxiliary": [{"MultiheadDetection": {"source": [1], "tags": "AUX", "output": True}}],
            },
        }
    )
    return YOLO(cfg, class_num=3)


def _prediction_sum(scales):
    return sum(cls.sum() + distribution.sum() for cls, distribution, _ in scales)


def test_one2one_gradients_are_isolated_from_shared_features():
    model = tiny_model().train()
    image = torch.randn(2, 3, 8, 8, requires_grad=True)
    output = model(image)
    assert set(output) == {"Main", "One2One", "AUX"}
    _prediction_sum(output["One2One"]).backward()
    main = model.model[1]
    assert image.grad is None
    assert all(parameter.grad is None for parameter in model.model[0].parameters())
    assert all(parameter.grad is None for parameter in main.heads.parameters())
    assert main.one2one_heads[0].class_conv[-1].weight.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    output = model(image, shortcut="Main")
    assert set(output) == {"Main", "One2One"}
    _prediction_sum(output["Main"]).backward()
    assert image.grad.abs().sum() > 0
    assert model.model[0].conv.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in main.one2one_heads.parameters())


@pytest.mark.parametrize("shortcut", [None, "Main"])
def test_eval_runs_only_one2one_and_skips_auxiliary(shortcut):
    model = tiny_model().eval()
    image = torch.randn(2, 3, 8, 8)
    main = model.model[1]
    with torch.no_grad():
        expected = main.one2one_heads[0](model.model[0](image))
    main.heads[0].forward = Mock(side_effect=AssertionError("dense head must not run"))
    model.model[2].forward = Mock(side_effect=AssertionError("auxiliary must not run"))
    with torch.no_grad():
        output = model(image, shortcut=shortcut)
    assert set(output) == {"Main"}
    for actual, reference in zip(output["Main"][0], expected):
        torch.testing.assert_close(actual, reference)


def test_baseline_keeps_original_heads_and_forward_contract():
    model = tiny_model(nms_free=False)
    assert not hasattr(model.model[1], "one2one_heads")
    assert not any("one2one" in key for key in model.state_dict())
    assert set(model(torch.randn(2, 3, 8, 8))) == {"Main", "AUX"}


@pytest.mark.parametrize("format", ["inner", "full", "lightning"])
def test_legacy_transfer_initializes_one2one_from_loaded_dense_weights(format):
    baseline = tiny_model(nms_free=False)
    baseline.model[1].heads[0].class_conv[-1].bias.data.fill_(0.75)
    expected = baseline.model[1].heads.state_dict()
    if format == "inner":
        weights = baseline.model.state_dict()
    elif format == "full":
        weights = baseline.state_dict()
    else:
        weights = {"state_dict": {f"model.{key}": value for key, value in baseline.state_dict().items()}}
    model = tiny_model()
    model.save_load_weights(weights)
    for key, actual in model.model[1].one2one_heads.state_dict().items():
        torch.testing.assert_close(actual, expected[key])
    assert model.model[1].heads[0].class_conv[-1].weight is not model.model[1].one2one_heads[0].class_conv[-1].weight


@pytest.mark.parametrize("format", ["inner", "full", "lightning"])
def test_trained_checkpoint_roundtrip_preserves_independent_one2one(format, tmp_path):
    source = tiny_model()
    source.model[1].one2one_heads[0].class_conv[-1].bias.data.fill_(2.5)
    if format == "inner":
        weights = source.model.state_dict()
    elif format == "full":
        weights = source.state_dict()
    else:
        weights = {"state_dict": {f"model.{key}": value for key, value in source.state_dict().items()}}
    checkpoint = tmp_path / "trained.pt"
    torch.save(weights, checkpoint)
    restored = tiny_model()
    restored.save_load_weights(checkpoint)
    for key, actual in restored.state_dict().items():
        torch.testing.assert_close(actual, source.state_dict()[key])
    assert not torch.equal(
        restored.model[1].heads[0].class_conv[-1].bias, restored.model[1].one2one_heads[0].class_conv[-1].bias
    )
    with pytest.raises(ValueError, match="model.nms_free=true"):
        tiny_model(nms_free=False).save_load_weights(checkpoint)


@pytest.mark.parametrize("mismatch", ["missing", "shape"])
def test_rejects_incomplete_trained_one2one_checkpoint(mismatch):
    model = tiny_model()
    weights = deepcopy(model.model.state_dict())
    name = "1.one2one_heads.0.class_conv.2.bias"
    if mismatch == "missing":
        del weights[name]
    else:
        weights[name] = torch.zeros(99)
    with pytest.raises(ValueError, match="missing or incompatible"):
        model.save_load_weights(weights)


def test_rejects_v7_detection_heads():
    with pytest.raises(ValueError, match="requires YOLOv9 Detection"):
        tiny_model(version="v7")


def test_rejects_non_detection_main_output():
    cfg = OmegaConf.create(
        {
            "name": "classification",
            "nms_free": True,
            "anchor": {"reg_max": 4},
            "model": {
                "backbone": [{"Conv": {"source": 0, "args": {"out_channels": 16, "kernel_size": 3}}}],
                "head": [{"Classification": {"source": 1, "tags": "Main", "output": True}}],
            },
        }
    )
    with pytest.raises(ValueError, match="Main MultiheadDetection output"):
        YOLO(cfg, class_num=3)


@pytest.mark.parametrize("nms_free", [False, True])
def test_qat_checkpoint_cannot_bypass_nms_free_guards(nms_free):
    checkpoint = {"qat": {}, "model_state_dict": tiny_model().state_dict()}
    expected = "floating-point checkpoints only" if nms_free else "model.nms_free=true"
    with pytest.raises(ValueError, match=expected):
        tiny_model(nms_free=nms_free).save_load_weights(checkpoint)


def test_create_model_rejects_nms_free_qat_before_loading_weights():
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "yolo/config/model/v9-t.yaml")
    cfg.nms_free = True
    qat_cfg = OmegaConf.create({"enabled": True})
    with pytest.raises(ValueError, match="floating-point training only"):
        create_model(cfg, weight_path=Path("must-not-download.pt"), qat_cfg=qat_cfg)


@pytest.mark.parametrize("value", ["false", 1])
def test_nms_free_requires_boolean_config(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        tiny_model(nms_free=value)


@pytest.mark.parametrize("family", ["v9", "v9-sr"])
@pytest.mark.parametrize("size", ["t", "s", "m", "c"])
def test_supported_models_nms_free_eval_shapes(family, size):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / f"yolo/config/model/{family}-{size}.yaml")
    cfg.nms_free = True
    model = create_model(cfg, weight_path=None, class_num=3).eval()
    with torch.inference_mode():
        outputs = model(torch.randn(1, 3, 64, 96))
    assert set(outputs) == {"Main"}
    assert len(outputs["Main"]) == 3
    for (classes, distribution, boxes), (height, width) in zip(outputs["Main"], [(8, 12), (4, 6), (2, 3)]):
        assert classes.shape == (1, 3, height, width)
        assert distribution.shape == (1, 16, 4, height, width)
        assert boxes.shape == (1, 4, height, width)
        assert all(torch.isfinite(tensor).all() for tensor in (classes, distribution, boxes))
