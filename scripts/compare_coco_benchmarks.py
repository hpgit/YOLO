"""Summarize paired subset benchmarks and fixed-weight prediction equivalence."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import torch


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--optimized", type=Path, nargs="+", required=True)
    parser.add_argument("--fixed-reference", type=Path, required=True)
    parser.add_argument("--fixed-optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads((path / "result.json").read_text())
    baseline, optimized = [read(p) for p in args.baseline], [read(p) for p in args.optimized]
    assert len(baseline) == len(optimized)
    for group in (baseline, optimized):
        assert all(r["source_sha256"] == group[0]["source_sha256"] for r in group)
    reference_config = (args.baseline[0] / "config.yaml").read_text()
    assert all((p / "config.yaml").read_text() == reference_config for p in args.baseline + args.optimized)
    caches = [r["dataset_cache_sha256"] for r in baseline + optimized if "dataset_cache_sha256" in r]
    if caches:
        assert all(cache == caches[0] for cache in caches)
    for result in baseline + optimized:
        for key in (
            "dataset_manifest_sha256",
            "image_size",
            "batch_size",
            "workers",
            "seed",
            "torch",
            "gpu",
            "train_images",
            "val_images",
        ):
            assert result[key] == baseline[0][key], key
    summary = {"runs": {"baseline": [str(p) for p in args.baseline], "optimized": [str(p) for p in args.optimized]}}
    for key in ("train_seconds", "validation_seconds", "fit_or_validate_seconds", "train_peak_allocated_bytes"):
        before, after = median(r[key] for r in baseline), median(r[key] for r in optimized)
        summary[key] = {
            "baseline_median": before,
            "optimized_median": after,
            "ratio_before_over_after": before / after,
            "reduction_percent": 100 * (1 - after / before),
        }
    old = torch.load(args.fixed_reference / "predictions.pt", weights_only=True)
    new = torch.load(args.fixed_optimized / "predictions.pt", weights_only=True)
    reference_result, fixed_result = read(args.fixed_reference), read(args.fixed_optimized)
    if caches:
        assert fixed_result["dataset_cache_sha256"] == caches[0]
    assert fixed_result["validation_only"]
    assert Path(fixed_result["eval_state"]).resolve() == (args.fixed_reference / "ema.pt").resolve()
    assert (
        fixed_result["eval_state_sha256"] == hashlib.sha256((args.fixed_reference / "ema.pt").read_bytes()).hexdigest()
    )
    for key in ("dataset_manifest_sha256", "image_size", "batch_size", "workers", "seed", "torch", "gpu", "val_images"):
        assert reference_result[key] == fixed_result[key], key
    assert len(old) == len(new) == reference_result["val_images"]
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    old_metrics, new_metrics = read(args.fixed_reference)["metrics"], read(args.fixed_optimized)["metrics"]
    metric_keys = [key for key in new_metrics if key.startswith(("map", "mar", "PyCOCO/"))]
    assert metric_keys
    for key in metric_keys:
        assert old_metrics[key] == new_metrics[key], (key, old_metrics[key], new_metrics[key])
    summary["fixed_weight_equivalence"] = {
        "reference": str(args.fixed_reference),
        "optimized": str(args.fixed_optimized),
        "images": len(old),
        "predictions_bitwise_equal": True,
        "metrics_exactly_equal": True,
        "metric_keys": metric_keys,
    }
    summary["cache_hashes_recorded_for_all_timed_runs"] = len(caches) == len(baseline + optimized)
    for a, b in zip(baseline, optimized):
        assert len(a["losses"]) == len(b["losses"]) > 0
    summary["paired_training_loss_max_abs_difference"] = [
        max(abs(x - y) for x, y in zip(a["losses"], b["losses"])) for a, b in zip(baseline, optimized)
    ]
    summary["paired_trained_ema_bitwise_equal"] = []
    for before_path, after_path in zip(args.baseline, args.optimized):
        before_state = torch.load(before_path / "ema.pt", weights_only=True, map_location="cpu")
        after_state = torch.load(after_path / "ema.pt", weights_only=True, map_location="cpu")
        identical = before_state.keys() == after_state.keys() and all(
            torch.equal(value, after_state[key]) for key, value in before_state.items()
        )
        summary["paired_trained_ema_bitwise_equal"].append(identical)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
