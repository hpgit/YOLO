from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize
from torch import nn

from yolo.model.yolo import create_model
from yolo.utils.bounding_box_utils import Anc2Box, Vec2Box

IMAGE_SIZE = [96, 64]
EXPECTED_STRIDES = [8, 16, 32]


def _anchor_config():
    return SimpleNamespace(anchor=[[[10, 13], [16, 30], [33, 23]]] * 3)


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _training_states(model):
    return [module.training for module in model.modules()]


class ProbeModel(nn.Module):
    def __init__(self, *, anchor_based=False, fail=False, device="cpu", dtype=torch.float32):
        super().__init__()
        self.num_classes = 3
        self.anchor_based = anchor_based
        self.fail = fail
        self.conv = nn.Conv2d(3, 3, 1, device=device, dtype=dtype)
        self.bn = nn.BatchNorm2d(3, device=device, dtype=dtype)
        self.nested = nn.Sequential(nn.Identity(), nn.Identity())
        self.observed = None

    def forward(self, x):
        self.observed = {
            "device": x.device,
            "dtype": x.dtype,
            "grad_enabled": torch.is_grad_enabled(),
            "inference_mode": torch.is_inference_mode_enabled(),
            "all_eval": all(not module.training for module in self.modules()),
        }
        if self.fail:
            raise RuntimeError("probe failed")

        outputs = []
        for stride in EXPECTED_STRIDES:
            feature = x.new_zeros((1, 8, x.shape[-2] // stride, x.shape[-1] // stride))
            outputs.append(feature if self.anchor_based else (feature, feature, feature))
        return {"Main": outputs}


@pytest.mark.parametrize("mode", ["train", "eval", "mixed"])
def test_vec2box_probe_restores_exact_module_modes_and_state(mode):
    model = ProbeModel()
    if mode == "eval":
        model.eval()
    elif mode == "mixed":
        model.train()
        model.bn.eval()
        model.nested[0].eval()

    modes_before = _training_states(model)
    state_before = _state(model)
    converter = Vec2Box(model, SimpleNamespace(), IMAGE_SIZE, torch.device("cpu"))

    assert converter.strides == EXPECTED_STRIDES
    assert _training_states(model) == modes_before
    assert all(torch.equal(value, state_before[name]) for name, value in model.state_dict().items())
    assert model.observed["all_eval"]
    assert not model.observed["grad_enabled"]
    assert model.observed["inference_mode"]


@pytest.mark.parametrize("converter_type", [Vec2Box, Anc2Box])
def test_probe_restores_module_modes_when_forward_raises(converter_type):
    model = ProbeModel(anchor_based=converter_type is Anc2Box, fail=True)
    model.train()
    model.bn.eval()
    modes_before = _training_states(model)

    config = _anchor_config() if converter_type is Anc2Box else SimpleNamespace()
    with pytest.raises(RuntimeError, match="probe failed"):
        converter_type(model, config, IMAGE_SIZE, torch.device("cpu"))

    assert _training_states(model) == modes_before


@pytest.mark.parametrize("converter_type", [Vec2Box, Anc2Box])
def test_probe_uses_model_dtype_and_disables_autograd(converter_type):
    model = ProbeModel(anchor_based=converter_type is Anc2Box, dtype=torch.float64)
    config = _anchor_config() if converter_type is Anc2Box else SimpleNamespace()

    converter = converter_type(model, config, IMAGE_SIZE, torch.device("cpu"))

    assert converter.strides == EXPECTED_STRIDES
    assert model.observed == {
        "device": torch.device("cpu"),
        "dtype": torch.float64,
        "grad_enabled": False,
        "inference_mode": True,
        "all_eval": True,
    }


@pytest.mark.parametrize("model_name", ["v9-t", "v9-s", "v9-m"])
def test_real_v9_probe_preserves_state_and_infers_rectangular_strides(model_name):
    with initialize(config_path="../../yolo/config", version_base=None):
        cfg = compose(config_name="config", overrides=[f"model={model_name}"])
    model = create_model(cfg.model, weight_path=False)
    model.train()
    next(module for module in model.modules() if isinstance(module, nn.BatchNorm2d)).eval()
    modes_before = _training_states(model)
    state_before = _state(model)

    converter = Vec2Box(model, cfg.model.anchor, IMAGE_SIZE, torch.device("cpu"))

    assert converter.strides == EXPECTED_STRIDES
    assert _training_states(model) == modes_before
    assert all(torch.equal(value, state_before[name]) for name, value in model.state_dict().items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_probe_uses_cuda_model_device():
    model = ProbeModel(device="cuda").train()

    converter = Vec2Box(model, SimpleNamespace(), IMAGE_SIZE, torch.device("cuda"))

    assert converter.strides == EXPECTED_STRIDES
    assert model.observed["device"].type == "cuda"
