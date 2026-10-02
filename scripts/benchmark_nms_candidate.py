"""Compare unchanged NMS outputs and timings on captured real COCO predictions."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from nms_grouping_candidate import grouped_batched_nms
from omegaconf import OmegaConf
from torchvision.ops import batched_nms as original_batched_nms

import yolo.utils.bounding_box_utils as bounding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "runs/performance-round2/nms-inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/performance-round2/nms-comparison.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    original = original_batched_nms
    configured = bounding.batched_nms
    cfg = OmegaConf.load(ROOT / "yolo/config/task/validation.yaml").nms
    totals = defaultdict(float)
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for path in sorted(args.inputs.glob("*.pt")):
            inputs = torch.load(path, weights_only=True, map_location="cuda")
            cls, boxes = inputs["cls"], inputs["boxes"]
            # Per-input warmup before the two measured, alternating comparisons.
            for function in (original, grouped_batched_nms):
                bounding.batched_nms = function
                bounding.bbox_nms(cls, boxes, cfg)
            times = defaultdict(float)
            for repeat in range(2):
                results = {}
                variants = [("baseline", original), ("grouped", grouped_batched_nms)]
                for label, function in variants[:: 1 if repeat == 0 else -1]:
                    bounding.batched_nms = function
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    results[label] = bounding.bbox_nms(cls, boxes, cfg)
                    torch.cuda.synchronize()
                    times[label] += time.perf_counter() - start
                for old, new in zip(results["baseline"], results["grouped"]):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
            for key, value in times.items():
                totals[key] += value
            rows.append({"input": path.name, "images": len(cls), "seconds_for_two_repeats": dict(times)})
    bounding.batched_nms = configured
    result = {
        "bitwise_equal": True,
        "images": sum(row["images"] for row in rows),
        "seconds_for_two_repeats": dict(totals),
        "speedup": totals["baseline"] / totals["grouped"],
        "batches": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
