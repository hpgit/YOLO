"""Reproducible real-COCO pose smoke test; no paper-quality metric claims.

Run from the repository root with the project environment:
  python scripts/smoke_pose.py --output runs/pose-smoke
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from hydra import compose, initialize_config_dir
from lightning import Callback, Trainer
from omegaconf import OmegaConf

from yolo.tools.solver import TrainModel
from yolo.utils.checkpoint_utils import YOLOCheckpoint
from yolo.utils.logging_utils import set_seed


class Progress(Callback):
    def __init__(self):
        self.losses = []
        self.started = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        value = float(outputs["loss"] if isinstance(outputs, dict) else outputs)
        if not math.isfinite(value):
            raise RuntimeError(f"Nonfinite smoke loss at batch {batch_idx}")
        self.losses.append(value)
        if batch_idx % 25 == 0:
            print(
                json.dumps(
                    {
                        "epoch": trainer.current_epoch,
                        "batch": batch_idx,
                        "batches": trainer.num_training_batches,
                        "loss": value,
                        "seconds": round(time.perf_counter() - self.started, 1),
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/pose-smoke")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty smoke output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "yolo/config"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=train-pose",
                "model=v9-t-pose",
                "dataset=coco-pose-smoke",
                "weight=false",
                f'dataset.path={root / "data/coco"}',
                f"image_size=[{args.size},{args.size}]",
                f"task.epoch={args.epochs}",
                f"task.data.batch_size={args.batch_size}",
                f"task.data.equivalent_batch_size={args.batch_size}",
                f"task.validation.data.batch_size={args.batch_size}",
                f"cpu_num={args.workers}",
                "use_wandb=false",
                "use_tensorboard=false",
                "accelerator=cpu",
                "device=1",
                "precision=32-true",
                "task.ema.enable=false",
                "task.validation.nms.max_bbox=20",
            ],
        )
    set_seed(cfg.lucky_number)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    model = TrainModel(cfg)
    manifests = {}
    for name, loader in [("train", model.train_loader), ("validation", model.val_loader)]:
        dataset = loader.dataset
        ids = list(dataset.image_ids)
        manifests[name] = {"count": len(ids), "image_ids": ids}
        split = cfg.dataset.train if name == "train" else cfg.dataset.validation
        annotation = Path(cfg.dataset.path) / "annotations" / f"person_keypoints_{split}.json"
        manifests[name]["annotation_sha256"] = hashlib.sha256(annotation.read_bytes()).hexdigest()
    (output / "subset_manifest.json").write_text(json.dumps(manifests, indent=2))
    # Track actual parameter updates in both bbox and pose output projections.
    tracked = {
        name: value.detach().cpu().clone()
        for name, value in model.model.named_parameters()
        if any(token in name for token in ("pose_logits.weight", "visibility_conv.weight", "anchor_conv.2.weight"))
    }
    progress = Progress()
    checkpoint = YOLOCheckpoint(output / "checkpoints")
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=args.epochs,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checkpoint, progress],
        default_root_dir=output,
        num_sanity_val_steps=0,
        deterministic=True,
    )
    started = time.perf_counter()
    trainer.fit(model)
    final_path = output / "pose-smoke.pt"
    torch.save({"weights": model.model.model.state_dict(), "pose_config": model.model.pose_config}, final_path)
    params = dict(model.model.named_parameters())
    deltas = {name: float((params[name].detach().cpu() - before).abs().max()) for name, before in tracked.items()}
    if not deltas or not all(math.isfinite(value) and value > 0 for value in deltas.values()):
        raise RuntimeError(f"Expected finite bbox and pose parameter updates: {deltas}")
    metrics = {str(name): float(value) for name, value in trainer.callback_metrics.items() if value.numel() == 1}
    report = {
        "purpose": "Execution smoke only; not a trained accuracy benchmark",
        "epochs": args.epochs,
        "image_size": [args.size, args.size],
        "device": "cpu",
        "batch_size": args.batch_size,
        "seed": cfg.lucky_number,
        "train_images": manifests["train"]["count"],
        "validation_images": manifests["validation"]["count"],
        "optimizer_steps": trainer.global_step,
        "training_batches": len(progress.losses),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "parameter_max_abs_updates": deltas,
        "checkpoint": str(final_path),
        "loss_min": min(progress.losses),
        "loss_max": max(progress.losses),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
