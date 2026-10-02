from collections import OrderedDict

import pytest
import torch

from yolo.utils import ema_utils
from yolo.utils.ema_utils import foreach_ema_update


@pytest.fixture(scope="module", autouse=True)
def _limit_cpu_threads():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 2))
    yield
    torch.set_num_threads(previous_threads)


def _reference_update(model_state, ema_state, decay):
    with torch.no_grad():
        for key, current in model_state.items():
            detached = current.detach()
            ema_state[key] = detached + (ema_state[key] - detached) * decay
    return ema_state


def _clone_state(state, *, requires_grad=False):
    cloned = OrderedDict()
    for key, value in state.items():
        copy = value.clone()
        if requires_grad and copy.is_floating_point():
            copy.requires_grad_()
        cloned[key] = copy
    return cloned


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_foreach_ema_is_exact_across_multiple_steps(dtype):
    initial = OrderedDict(
        weight=torch.tensor([[1.25, -2.5, 4.0], [0.125, 8.0, -0.75]], dtype=dtype),
        bias=torch.tensor([3.0, -1.5, 0.25], dtype=dtype),
        num_batches_tracked=torch.tensor(7, dtype=torch.int64),
    )
    expected = _clone_state(initial)
    actual = _clone_state(initial)

    for step, decay in enumerate((0.0, 0.37, 0.8125, 0.9999), start=1):
        model_state = OrderedDict(
            weight=torch.tensor(
                [[step * 0.3, -step * 1.1, step + 0.0625], [2.25 - step, step * 3.5, -0.2 * step]],
                dtype=dtype,
                requires_grad=True,
            ),
            bias=torch.tensor([step + 0.5, -step * 0.75, step / 3], dtype=dtype, requires_grad=True),
            num_batches_tracked=torch.tensor(7 + step, dtype=torch.int64),
        )

        expected_result = _reference_update(model_state, expected, decay)
        actual_result = foreach_ema_update(model_state, actual, decay)

        assert actual_result is actual
        assert torch.get_num_threads() <= 2
        assert actual["num_batches_tracked"].dtype == torch.float32
        for key in model_state:
            torch.testing.assert_close(actual[key], expected_result[key], rtol=0, atol=0)
            assert not actual[key].requires_grad
            assert actual[key].untyped_storage().data_ptr() != model_state[key].untyped_storage().data_ptr()
            assert model_state[key].grad is None


def test_integer_counter_matches_initial_and_later_mixed_dtype_updates():
    expected = OrderedDict(counter=torch.tensor(11, dtype=torch.int64))
    actual = _clone_state(expected)

    first_model_state = OrderedDict(counter=torch.tensor(13, dtype=torch.int64))
    _reference_update(first_model_state, expected, 0.25)
    foreach_ema_update(first_model_state, actual, 0.25)
    assert actual["counter"].dtype == expected["counter"].dtype == torch.float32
    torch.testing.assert_close(actual["counter"], expected["counter"], rtol=0, atol=0)

    second_model_state = OrderedDict(counter=torch.tensor(21, dtype=torch.int64))
    _reference_update(second_model_state, expected, 0.625)
    foreach_ema_update(second_model_state, actual, 0.625)
    assert actual["counter"].dtype == expected["counter"].dtype == torch.float32
    torch.testing.assert_close(actual["counter"], expected["counter"], rtol=0, atol=0)


def test_update_disables_grad_and_does_not_alias_model_storage():
    model_state = OrderedDict(
        first=torch.tensor([1.0, 2.0], requires_grad=True),
        second=torch.tensor([-3.0, 5.0], requires_grad=True),
    )
    ema_state = _clone_state(model_state, requires_grad=True)

    foreach_ema_update(model_state, ema_state, 0.9)

    for key, current in model_state.items():
        assert not ema_state[key].requires_grad
        assert current.grad is None
        assert ema_state[key].untyped_storage().data_ptr() != current.untyped_storage().data_ptr()


def test_non_contiguous_tensors_match_original_expression():
    current = torch.arange(24, dtype=torch.float64).reshape(4, 6).transpose(0, 1)
    initial = torch.linspace(-3.0, 5.0, 24, dtype=torch.float64).reshape(4, 6).transpose(0, 1)
    assert not current.is_contiguous()
    assert not initial.is_contiguous()

    model_state = OrderedDict(weight=current)
    expected = OrderedDict(weight=initial.clone())
    actual = OrderedDict(weight=initial.clone())

    _reference_update(model_state, expected, 0.73)
    foreach_ema_update(model_state, actual, 0.73)

    torch.testing.assert_close(actual["weight"], expected["weight"], rtol=0, atol=0)


def test_incompatible_tensor_pair_uses_scalar_fallback(monkeypatch):
    model_state = OrderedDict(weight=torch.tensor([1.0, -4.0, 7.5], requires_grad=True))
    expected = OrderedDict(weight=torch.tensor([3.0, 2.0, -1.0]))
    actual = _clone_state(expected, requires_grad=True)

    monkeypatch.setattr(ema_utils, "_is_foreach_compatible", lambda ema, current: False)

    def fail_if_called(*args, **kwargs):
        pytest.fail("foreach should not run for an incompatible tensor pair")

    monkeypatch.setattr(ema_utils, "_FOREACH_SUB", fail_if_called)
    _reference_update(model_state, expected, 0.61)
    foreach_ema_update(model_state, actual, 0.61)

    torch.testing.assert_close(actual["weight"], expected["weight"], rtol=0, atol=0)
    assert not actual["weight"].requires_grad
    assert model_state["weight"].grad is None


def test_foreach_out_of_memory_error_is_not_swallowed(monkeypatch):
    model_state = OrderedDict(weight=torch.tensor([1.0, 2.0]))
    ema_state = OrderedDict(weight=torch.tensor([3.0, 4.0]))
    original = ema_state["weight"]

    def raise_out_of_memory(*args, **kwargs):
        raise torch.OutOfMemoryError("test allocation failure")

    monkeypatch.setattr(ema_utils, "_FOREACH_SUB", raise_out_of_memory)
    with pytest.raises(torch.OutOfMemoryError, match="test allocation failure"):
        foreach_ema_update(model_state, ema_state, 0.9)

    assert ema_state["weight"] is original


def test_key_mismatch_fails_before_mutating_ema_state():
    model_state = OrderedDict(weight=torch.tensor([1.0]))
    ema_state = OrderedDict(other=torch.tensor([2.0]))
    original = ema_state["other"]

    with pytest.raises(KeyError, match="state dict keys differ"):
        foreach_ema_update(model_state, ema_state, 0.9)

    assert ema_state["other"] is original
