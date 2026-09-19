"""Validate once and replay the exact exported detections through reference COCOeval.

This checks evaluation parity, not model/paper accuracy reproduction. Existing
output directories are never overwritten. Use --dataset for a diagnostic subset.
"""

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning import Callback, Trainer, seed_everything
from omegaconf import OmegaConf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolo.tools.solver import ValidateModel
from yolo.utils.coco_eval import CocoJsonEvaluator

NAMES = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Capture(Callback):
    def __init__(self):
        self.batches = []

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        self.batches.append(
            {
                "predictions": [p.detach().cpu() for p in outputs[0]],
                "paths": [str(p) for p in batch[4]],
                "image_size": [batch[1].shape[3], batch[1].shape[2]],
            }
        )
        if batch_idx % 50 == 0:
            print(
                f"Validation: batch {batch_idx + 1}, captured {sum(len(b['paths']) for b in self.batches)} images",
                flush=True,
            )

    def on_validation_epoch_end(self, trainer, module):
        self.predictions = module.metric.predictions
        self.image_ids = module.metric.image_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/coco")
    parser.add_argument("--model", default="v9-c")
    parser.add_argument("--weight", type=Path, default=ROOT / "weights/v9-c.pt")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(10, workers=True)
    torch.set_num_threads(4)
    annotation = args.dataset.resolve() / "annotations/instances_val2017.json"
    annotation_hash = sha256(annotation)
    weight_hash = sha256(args.weight)
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "yolo").rglob("*.py"))}
    with initialize_config_dir(config_dir=str(ROOT / "yolo/config"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=validation",
                "task.evaluator=coco",
                "dataset=coco",
                "dataset.auto_download=null",
                f"dataset.path={args.dataset.resolve()}",
                f"model={args.model}",
                f"weight={args.weight.resolve()}",
                f"image_size=[{args.image_size},{args.image_size}]",
                f"cpu_num={args.workers}",
                f"task.data.batch_size={args.batch_size}",
                "use_wandb=false",
                "device=1",
            ],
        )
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    capture = Capture()
    model = ValidateModel(cfg)
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision="16-mixed" if args.accelerator == "gpu" else "32-true",
        callbacks=[capture],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        default_root_dir=args.output,
    )
    started = time.monotonic()
    measured = trainer.validate(model, verbose=False)[0]
    torch.save(capture.batches, args.output / "batches.pt")
    (args.output / "predictions.json").write_text(json.dumps(capture.predictions) + "\n")
    (args.output / "image_ids.json").write_text(json.dumps(capture.image_ids) + "\n")
    print(
        f"Inference complete: {len(capture.image_ids)} images. Evaluating saved JSON with reference COCOeval.",
        flush=True,
    )

    # Independent reference entry point: same API used in the original repository.
    # This deliberately does not call the new evaluator's compute implementation.
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
        dt = gt.loadRes(str(args.output / "predictions.json")) if capture.predictions else COCO()
        if not capture.predictions:
            dt.dataset = {"images": list(gt.imgs.values()), "categories": list(gt.cats.values()), "annotations": []}
            dt.createIndex()
        reference = COCOeval(gt, dt, "bbox")
        reference.params.imgIds = capture.image_ids
        reference.evaluate()
        reference.accumulate()
        reference.summarize()
    actual = np.array([measured[name] for name in NAMES])
    delta = np.abs(actual - reference.stats)
    np.testing.assert_allclose(actual, reference.stats, rtol=0, atol=1e-12)
    assert set(capture.image_ids) == set(gt.getImgIds()), "Validation did not cover every annotation image."
    print(f"Reference parity max error: {delta.max()}. Replaying saved batches in reverse image order.", flush=True)

    # Replay the same model outputs with different update boundaries/order; do
    # not regenerate model outputs, which would conflate evaluation and inference.
    replay = CocoJsonEvaluator(annotation, image_root=args.dataset.resolve() / "images/val2017")
    for batch in reversed(capture.batches):
        for prediction, path in zip(batch["predictions"], batch["paths"]):
            replay.update([prediction], [path], batch["image_size"])
    replay_stats = replay.compute()
    replay_delta = max(abs(float(replay_stats[n]) - float(reference.stats[i])) for i, n in enumerate(NAMES))
    assert replay_delta <= 1e-12, replay_delta
    assert annotation_hash == sha256(annotation), "Annotations changed during validation"
    assert weight_hash == sha256(args.weight), "Weights changed during validation"
    assert all(
        sha256(ROOT / path) == digest for path, digest in source_hashes.items()
    ), "Source changed during validation"
    result = {
        "status": "passed",
        "scope": "Same-prediction COCO evaluation parity; not paper AP reproduction",
        "images": len(capture.image_ids),
        "detections": len(capture.predictions),
        "model": args.model,
        "weight": str(args.weight.resolve()),
        "weight_sha256": weight_hash,
        "annotation": str(annotation),
        "annotation_sha256": annotation_hash,
        "image_size": args.image_size,
        "inference_batch_size": args.batch_size,
        "metrics": {
            n: {"integrated": float(actual[i]), "reference": float(reference.stats[i])} for i, n in enumerate(NAMES)
        },
        "max_abs_difference": float(delta.max()),
        "reordered_single_image_replay_max_abs_difference": replay_delta,
        "elapsed_seconds": time.monotonic() - started,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hashes,
        "versions": {name: version(name) for name in ("torch", "lightning", "pycocotools")},
    }
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
