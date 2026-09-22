from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch import nn

from yolo.model.module import AConv2, ADown2, Conv, FixedKernelConv2d, RepConv
from yolo.model.yolo import create_model
from yolo.utils.model_utils import EMA, create_optimizer

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "yolo/config/model"
KERNEL = torch.tensor([[5, 27, 5], [27, 127, 27], [5, 27, 5]], dtype=torch.float32) / 255


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_fixed_kernel_channel_isolation_and_padding():
    smooth = FixedKernelConv2d(4)
    x = torch.zeros(1, 4, 7, 9)
    x[0, 2, 3, 4] = 1
    expected = torch.zeros_like(x)
    expected[0, 2, 2:5, 3:6] = KERNEL
    torch.testing.assert_close(smooth(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(smooth.weight, KERNEL.expand(4, 1, 3, 3), rtol=0, atol=0)
    # Check boundary handling with independent single-channel convolutions.
    x = torch.randn(2, 4, 7, 9)
    expected = torch.cat([F.conv2d(c, KERNEL[None, None], padding=1) for c in x.split(1, 1)], 1)
    torch.testing.assert_close(smooth(x), expected)


@pytest.mark.parametrize("block_cls", [AConv2, ADown2])
@pytest.mark.parametrize("shape", [(16, 20), (15, 19)])
@pytest.mark.parametrize("optimizer_cls", [torch.optim.SGD, torch.optim.AdamW])
def test_smoothing_stays_frozen_through_training_and_reload(block_cls, shape, optimizer_cls):
    block = block_cls(4, 8).train()
    block.requires_grad_(True)
    fixed = block.avg_pool.weight.clone()
    assert not block.avg_pool.weight.requires_grad
    assert "avg_pool.weight" not in dict(block.named_parameters())
    assert "avg_pool.weight" not in block.state_dict()  # EMA must not average it.
    optimizer = optimizer_cls(block.parameters(), lr=0.01, weight_decay=0.1)
    trainable = next(block.parameters())
    before = trainable.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        x = torch.randn(2, 4, *shape, requires_grad=True)
        output = block(x)
        assert output.shape == (2, 8, (shape[0] + 1) // 2, (shape[1] + 1) // 2)
        output.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        optimizer.step()
        assert block.avg_pool.weight.grad is None
        assert torch.equal(block.avg_pool.weight, fixed)
    assert not torch.equal(trainable, before)

    restored = block_cls(4, 8)
    restored.load_state_dict(deepcopy(block.state_dict()))
    block.eval()
    restored.eval()
    torch.testing.assert_close(restored(x.detach()), block(x.detach()), rtol=0, atol=0)
    assert torch.equal(restored.avg_pool.weight, fixed)
    assert block.double().avg_pool.weight.dtype == torch.float64
    assert block(x.detach().double()).dtype == torch.float64


@pytest.mark.parametrize("size", ["t", "s", "m", "c"])
def test_sr_config_forward_backward_optimizer_and_ema(size):
    with initialize(config_path="../../yolo/config", version_base=None):
        cfg = compose(config_name="config", overrides=[f"model=v9-sr-{size}", "weight=null", "task=train"])
    baseline = OmegaConf.to_container(OmegaConf.load(CONFIG_ROOT / f"v9-{size}.yaml"))
    expected = deepcopy(baseline)
    expected["name"] = f"v9-sr-{size}"
    expected["activation"] = "Hardswish"
    for layers in expected["model"].values():
        for layer in layers:
            name = next(iter(layer))
            if name in {"AConv", "ADown"}:
                layer[name + "2"] = layer.pop(name)
            if size == "m":
                args = next(iter(layer.values())).get("args", {})
                for key in ("out_channels", "part_channels"):
                    value = args.get(key)
                    widths = {184: 192, 240: 256, 360: 384}
                    if key == "part_channels":
                        widths[480] = 512
                    if isinstance(value, list):
                        args[key] = [widths.get(channel, channel) for channel in value]
                    elif value is not None:
                        args[key] = widths.get(value, value)
    assert OmegaConf.to_container(cfg.model) == expected

    model = create_model(cfg.model, class_num=3, weight_path=None).train()
    blocks = [module for module in model.modules() if isinstance(module, (AConv2, ADown2))]
    assert len(blocks) == (5 if size in {"t", "s"} else 8)
    assert all(type(block) is (ADown2 if size == "c" else AConv2) for block in blocks)
    assert not any(isinstance(module, (nn.SiLU, nn.AvgPool2d)) for module in model.modules())
    for module in model.modules():
        if isinstance(module, (Conv, RepConv)):
            assert isinstance(module.act, (nn.Hardswish, nn.Identity))
        if isinstance(module, RepConv):
            assert isinstance(module.conv1.act, nn.Identity)
            assert isinstance(module.conv2.act, nn.Identity)
    for tag in ("Main", "AUX"):
        head = model.model[model.layer_index[tag] - 1]
        assert all(isinstance(module.act, nn.Hardswish) for module in head.modules() if isinstance(module, Conv))

    optimizer = create_optimizer(model, cfg.task.optimizer)
    before = model.model[0].conv.weight.detach().clone()
    fixed = [block.avg_pool.weight.clone() for block in blocks]
    output = model(torch.randn(2, 3, 64, 96))
    assert set(output) == {"Main", "AUX"}
    for scales in output.values():
        for (cls, anchor, box), (h, w) in zip(scales, [(8, 12), (4, 6), (2, 3)]):
            assert cls.shape == (2, 3, h, w)
            assert anchor.shape == (2, 16, 4, h, w)
            assert box.shape == (2, 4, h, w)
            assert all(torch.isfinite(tensor).all() for tensor in (cls, anchor, box))
    loss = sum(tensor.square().mean() for scales in output.values() for scale in scales for tensor in scale)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(model.model[0].conv.weight, before)
    assert all(torch.equal(block.avg_pool.weight, weight) for block, weight in zip(blocks, fixed))

    module = SimpleNamespace(model=model)
    ema = EMA()
    ema.setup(None, module, "fit")
    ema.update(module)
    ema.on_validation_start(None, module)
    ema_fixed = [layer.weight for layer in module.ema.modules() if isinstance(layer, FixedKernelConv2d)]
    assert all(torch.equal(actual, expected) for actual, expected in zip(ema_fixed, fixed))


@pytest.mark.parametrize("class_num", [3, 80])
def test_sr_m_convolution_inputs_are_aligned_without_shrinking(class_num):
    model = create_model(OmegaConf.load(CONFIG_ROOT / "v9-sr-m.yaml"), class_num=class_num, weight_path=None).eval()
    original = create_model(OmegaConf.load(CONFIG_ROOT / "v9-m.yaml"), class_num=class_num, weight_path=None)
    original_convs = {name: module for name, module in original.named_modules() if isinstance(module, nn.Conv2d)}
    observed = {}
    handles = []

    def record_input(name):
        def hook(module, inputs):
            observed[name] = inputs[0].shape[1]

        return hook

    rgb_stems = {f"model.{index}.conv" for index, layer in enumerate(model.model) if layer.source == 0}
    assert len(rgb_stems) == 2  # Main and AUX RGB stems.
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            baseline = original_convs.pop(name)
            assert module.in_channels >= baseline.in_channels, name
            assert module.out_channels >= baseline.out_channels, name
            if name in rgb_stems:
                assert module.in_channels == 3
            else:
                assert module.in_channels % 32 == 0, (name, module.in_channels)
            handles.append(module.register_forward_pre_hook(record_input(name)))
        elif isinstance(module, FixedKernelConv2d):
            assert module.groups % 32 == 0, name
            handles.append(module.register_forward_pre_hook(record_input(name)))
        elif isinstance(module, nn.Conv3d):
            assert name.endswith("anc2vec.anc2vec") and module.in_channels == 16
    assert not original_convs

    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 64, 96))
    finally:
        for handle in handles:
            handle.remove()
    assert len(observed) == len(handles)
    assert all(channels == 3 if name in rgb_stems else channels % 32 == 0 for name, channels in observed.items())


def test_sr_activation_does_not_change_original_model_defaults():
    sr_cfg = OmegaConf.load(CONFIG_ROOT / "v9-sr-t.yaml")
    create_model(sr_cfg, weight_path=None)
    original = create_model(OmegaConf.load(CONFIG_ROOT / "v9-t.yaml"), weight_path=None)
    assert isinstance(original.model[0].act, nn.SiLU)
    assert any(isinstance(module, nn.AvgPool2d) for module in original.modules())
    assert not any(isinstance(module, nn.Hardswish) for module in original.modules())
