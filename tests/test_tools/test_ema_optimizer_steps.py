from copy import deepcopy
from math import exp
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from yolo.utils.model_utils import EMA


class FloatAndIntegerState(nn.Module):
    def __init__(self, *, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0], device=device))
        self.register_buffer("counter", torch.tensor(3, dtype=torch.int64, device=device))

    def forward(self, value):
        return self.weight * value


class OverflowSkippingOptimizer(torch.optim.Optimizer):
    """Minimal fused-like optimizer whose raw step runs while overflow skips work."""

    def __init__(self, parameters, lr=0.25):
        super().__init__(parameters, {"lr": lr})
        self.found_inf = torch.tensor(0.0)

    @torch.no_grad()
    def step(self, closure=None):
        if bool(self.found_inf):
            return None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-group["lr"])
        return None


def _module(model):
    return SimpleNamespace(model=model)


def _start_callback(model, *, decay=0.9, tau=2.0, world_size=1):
    module = _module(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    trainer = SimpleNamespace(optimizers=[optimizer], world_size=world_size, sanity_checking=False)
    callback = EMA(decay=decay, tau=tau)
    callback.setup(trainer, module, "fit")
    callback.on_train_start(trainer, module)
    return callback, trainer, module, optimizer


def _optimizer_update(model, optimizer, gradient=4.0):
    optimizer.zero_grad(set_to_none=True)
    model.weight.grad = torch.tensor([gradient], device=model.weight.device)
    optimizer.step()


@pytest.mark.parametrize("run_sanity_validation", [False, True])
def test_ema_initializes_with_or_without_sanity_validation(run_sanity_validation):
    model = FloatAndIntegerState()
    module = _module(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(optimizers=[optimizer], world_size=1, sanity_checking=run_sanity_validation)
    callback = EMA()
    initial = deepcopy(model.state_dict())

    callback.setup(trainer, module, "fit")
    if run_sanity_validation:
        callback.on_validation_start(trainer, module)
    else:
        assert callback.ema_state_dict is None

    trainer.sanity_checking = False
    callback.on_train_start(trainer, module)

    assert callback.step == 0
    for key in initial:
        torch.testing.assert_close(callback.ema_state_dict[key], initial[key], rtol=0, atol=0)
    callback.teardown(trainer, module, "fit")


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_post_step_ema_matches_float_reference_and_keeps_integer_state(world_size):
    decay = 0.9
    tau = 2.0
    model = FloatAndIntegerState()
    callback, trainer, module, optimizer = _start_callback(model, decay=decay, tau=tau, world_size=world_size)
    initial_weight = callback.ema_state_dict["weight"].clone()
    initial_counter = callback.ema_state_dict["counter"].clone()
    model.counter.fill_(17)

    _optimizer_update(model, optimizer)

    step_decay = decay * (1 - exp(-1 / tau))
    expected_weight = model.weight.detach() + (initial_weight - model.weight.detach()) * step_decay
    assert callback.step == 1
    assert callback.tau == tau
    torch.testing.assert_close(callback.ema_state_dict["weight"], expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(callback.ema_state_dict["counter"], initial_counter, rtol=0, atol=0)
    callback.teardown(trainer, module, "fit")


def test_world_size_does_not_change_tau_or_ema_values():
    results = []
    for world_size in (1, 4):
        model = FloatAndIntegerState()
        callback, trainer, module, optimizer = _start_callback(model, world_size=world_size)
        _optimizer_update(model, optimizer)
        results.append((callback.tau, deepcopy(callback.ema_state_dict)))
        callback.teardown(trainer, module, "fit")

    assert results[0][0] == results[1][0] == 2.0
    for key in results[0][1]:
        torch.testing.assert_close(results[0][1][key], results[1][1][key], rtol=0, atol=0)


def test_no_optimizer_step_leaves_ema_unchanged():
    model = FloatAndIntegerState()
    callback, trainer, module, optimizer = _start_callback(model)
    initial = deepcopy(callback.ema_state_dict)

    model.weight.data.add_(10)
    model.counter.add_(5)
    optimizer.zero_grad(set_to_none=True)

    assert callback.step == 0
    for key in initial:
        torch.testing.assert_close(callback.ema_state_dict[key], initial[key], rtol=0, atol=0)
    callback.teardown(trainer, module, "fit")


def test_optimizer_found_inf_marker_skips_post_step_ema_update():
    model = FloatAndIntegerState()
    module = _module(model)
    optimizer = OverflowSkippingOptimizer(model.parameters())
    trainer = SimpleNamespace(optimizers=[optimizer], world_size=1, sanity_checking=False)
    callback = EMA(decay=0.9, tau=2.0)
    callback.setup(trainer, module, "fit")
    callback.on_train_start(trainer, module)
    initial = deepcopy(callback.ema_state_dict)
    optimizer.found_inf.fill_(1.0)

    _optimizer_update(model, optimizer)

    assert callback.step == 0
    for key in initial:
        torch.testing.assert_close(callback.ema_state_dict[key], initial[key], rtol=0, atol=0)

    optimizer.found_inf.zero_()
    _optimizer_update(model, optimizer)
    assert callback.step == 1
    callback.teardown(trainer, module, "fit")


def test_callback_state_restores_update_counter_and_ema_values():
    model = FloatAndIntegerState()
    callback, trainer, module, optimizer = _start_callback(model)
    _optimizer_update(model, optimizer)
    saved = deepcopy(callback.state_dict())
    callback.teardown(trainer, module, "fit")

    restored_model = FloatAndIntegerState()
    restored_module = _module(restored_model)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.25)
    restored_trainer = SimpleNamespace(optimizers=[restored_optimizer], world_size=8, sanity_checking=False)
    restored = EMA(decay=callback.decay, tau=callback.tau)
    restored.load_state_dict(saved)
    restored.setup(restored_trainer, restored_module, "fit")
    restored.on_train_start(restored_trainer, restored_module)

    assert restored.step == callback.step == 1
    for key in saved["ema_state_dict"]:
        torch.testing.assert_close(restored.ema_state_dict[key], saved["ema_state_dict"][key], rtol=0, atol=0)

    initial_ema = restored.ema_state_dict["weight"].clone()
    _optimizer_update(restored_model, restored_optimizer)
    expected_decay = restored.decay * (1 - exp(-2 / restored.tau))
    expected = restored_model.weight.detach() + (initial_ema - restored_model.weight.detach()) * expected_decay
    assert restored.step == 2
    torch.testing.assert_close(restored.ema_state_dict["weight"], expected, rtol=0, atol=0)
    restored.teardown(restored_trainer, restored_module, "fit")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_grad_scaler_overflow_skips_ema_update():
    model = FloatAndIntegerState(device="cuda")
    callback, trainer, module, optimizer = _start_callback(model)
    scaler = torch.amp.GradScaler("cuda")

    optimizer.zero_grad(set_to_none=True)
    finite_loss = model(torch.ones(1, device="cuda")).sum()
    scaler.scale(finite_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert callback.step == 1

    weight_before = model.weight.detach().clone()
    ema_before = deepcopy(callback.ema_state_dict)
    optimizer.zero_grad(set_to_none=True)
    overflow_loss = model(torch.full((1,), float("inf"), device="cuda")).sum()
    scaler.scale(overflow_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert callback.step == 1
    torch.testing.assert_close(model.weight, weight_before, rtol=0, atol=0)
    for key in ema_before:
        torch.testing.assert_close(callback.ema_state_dict[key], ema_before[key], rtol=0, atol=0)
    callback.teardown(trainer, module, "fit")
