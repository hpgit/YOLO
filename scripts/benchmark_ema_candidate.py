"""CUDA EMA timing and exactness on the real v9-t state tensor shapes."""

import json
import time
from math import exp
from pathlib import Path

import torch
from ema_foreach_candidate import foreach_ema_update


def scalar(model_state, ema_state, decay):
    for key, current in model_state.items():
        ema_state[key] = current + (ema_state[key] - current) * decay


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    model = torch.load(root / "runs/performance/baseline-1/ema.pt", weights_only=True, map_location="cuda")
    old = {key: value.clone() for key, value in model.items()}
    new = {key: value.clone() for key, value in model.items()}
    timings = {"scalar": 0.0, "foreach": 0.0}
    with torch.no_grad():
        for step in range(35):
            for value in model.values():
                value.add_(0.001 if value.is_floating_point() else 1)
            decay = 0.9999 * (1 - exp(-(step + 1) / 2000))
            variants = [("scalar", scalar, old), ("foreach", foreach_ema_update, new)]
            for name, function, state in variants[:: 1 if step % 2 == 0 else -1]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                function(model, state, decay)
                torch.cuda.synchronize()
                if step >= 5:
                    timings[name] += time.perf_counter() - start
            if step in (0, 1, 34):
                for key in old:
                    torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    result = {
        "bitwise_equal": True,
        "state_tensors": len(model),
        "timed_updates": 30,
        "milliseconds_per_update": {k: v / 30 * 1000 for k, v in timings.items()},
        "speedup": timings["scalar"] / timings["foreach"],
    }
    (root / "runs/performance-round2/ema-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
