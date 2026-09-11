"""Utilities for updating exponential moving-average model state."""

from collections import defaultdict
from typing import Callable, Dict, List, Mapping, MutableMapping, Tuple

import torch
from torch import Tensor


def _resolve_foreach(name: str) -> Callable:
    """Prefer a public foreach API if PyTorch adds one, with 2.6 fallback."""
    operation = getattr(torch, f"foreach_{name}", None)
    if operation is None:
        operation = getattr(torch, f"_foreach_{name}")
    return operation


_FOREACH_SUB = _resolve_foreach("sub")
_FOREACH_MUL = _resolve_foreach("mul")
_FOREACH_ADD = _resolve_foreach("add")

_GroupKey = Tuple[torch.device, torch.dtype, torch.dtype]
_GroupEntry = Tuple[str, Tensor, Tensor]


def _is_foreach_compatible(ema: Tensor, current: Tensor) -> bool:
    return (
        ema.layout == torch.strided
        and current.layout == torch.strided
        and ema.device == current.device
        and ema.device.type in ("cpu", "cuda")
    )


def _scalar_update(current: Tensor, ema: Tensor, decay: float) -> Tensor:
    return current.detach() + (ema - current.detach()) * decay


@torch.no_grad()
def foreach_ema_update(
    model_state_dict: Mapping[str, Tensor],
    ema_state_dict: MutableMapping[str, Tensor],
    decay: float,
) -> MutableMapping[str, Tensor]:
    """Update ``ema_state_dict`` while preserving the original EMA arithmetic.

    Dense CPU and CUDA tensors are grouped for foreach execution. Other
    layouts and device types use the original scalar expression.
    """
    missing = [key for key in model_state_dict if key not in ema_state_dict]
    unexpected = [key for key in ema_state_dict if key not in model_state_dict]
    if missing or unexpected:
        raise KeyError(f"state dict keys differ; missing EMA keys={missing}, unexpected EMA keys={unexpected}")

    groups: Dict[_GroupKey, List[_GroupEntry]] = defaultdict(list)
    scalar_entries: List[_GroupEntry] = []
    for key, current in model_state_dict.items():
        current = current.detach()
        ema = ema_state_dict[key]
        entry = (key, current, ema)
        if _is_foreach_compatible(ema, current):
            groups[(ema.device, ema.dtype, current.dtype)].append(entry)
        else:
            scalar_entries.append(entry)

    updates: Dict[str, Tensor] = {}
    for entries in groups.values():
        keys = [entry[0] for entry in entries]
        currents = [entry[1] for entry in entries]
        ema_values = [entry[2] for entry in entries]
        try:
            differences = _FOREACH_SUB(ema_values, currents)
            scaled_differences = _FOREACH_MUL(differences, decay)
            updated_values = _FOREACH_ADD(currents, scaled_differences)
        except torch.OutOfMemoryError:
            raise
        except (RuntimeError, TypeError, NotImplementedError):
            updated_values = [_scalar_update(current, ema, decay) for _, current, ema in entries]
        updates.update(zip(keys, updated_values))

    for key, current, ema in scalar_entries:
        updates[key] = _scalar_update(current, ema, decay)

    for key in model_state_dict:
        ema_state_dict[key] = updates[key]
    return ema_state_dict
