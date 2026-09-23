"""Training snapshots, inference weights, and named-run resumption."""

from math import isfinite
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from yolo.utils.logger import logger


class YOLOCheckpoint(ModelCheckpoint):
    """Keep the latest full snapshot and the best validated detector weights."""

    def __init__(self, dirpath):
        super().__init__(
            dirpath=dirpath,
            filename="epoch{epoch:04d}-step{step:08d}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_on_train_epoch_end=True,
        )
        self.best_map = float("-inf")

    def on_train_epoch_end(self, trainer, pl_module):
        # Manual accumulation can finish an epoch without an optimizer step.
        # Still snapshot the newer epoch and scheduler/validation state.
        self._last_global_step_saved = -1
        super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if trainer.state.fn == "fit" and not trainer.sanity_checking and "map" in trainer.callback_metrics:
            score = float(trainer.callback_metrics["map"])
            if isfinite(score) and score > self.best_map:
                if trainer.is_global_zero:
                    # Validation uses EMA when enabled; the inner state dict is
                    # the same weight-only format accepted by YOLO.save_load_weights.
                    detector = pl_module.ema
                    weights = {key: value.detach().cpu() for key, value in detector.model.state_dict().items()}
                    if getattr(detector, "pose_config", None) is not None:
                        weights = {"weights": weights, "pose_config": dict(detector.pose_config)}
                    destination = Path(self.dirpath) / "best.pt"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".pt.tmp")
                    torch.save(weights, temporary)
                    temporary.replace(destination)
                self.best_map = score
        super().on_validation_end(trainer, pl_module)

    def state_dict(self):
        return {**super().state_dict(), "best_map": self.best_map}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.best_map = state_dict.get("best_map", float("-inf"))


def latest_checkpoint(directory):
    """Order by saved epoch and global step, including legacy names and last.ckpt."""
    candidates = []
    for path in sorted(Path(directory).rglob("*.ckpt")):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            progress = (int(checkpoint["epoch"]), int(checkpoint["global_step"]))
            del checkpoint
        except Exception as error:
            logger.warning(f"Skipping unreadable training checkpoint {path}: {error}")
            continue
        candidates.append((*progress, str(path)))
    return Path(max(candidates)[2]) if candidates else None


def resolve_training_checkpoint(cfg, *, weight_explicit=False):
    """Explicit weights win; otherwise resume the furthest snapshot in this run."""
    if cfg.task.task != "train":
        return None
    weight = cfg.weight
    if isinstance(weight, (str, Path)) and weight:
        path = Path(weight).expanduser()
        if path.suffix == ".ckpt":
            if not path.is_file():
                raise FileNotFoundError(f"Training checkpoint does not exist: {path}")
            return path.resolve()
        return None
    # False permits named-run resume, falling back to random initialization.
    # Explicit True/null still request fresh pretrained/random initialization.
    if (weight_explicit and weight is not False) or weight is None or not cfg.name:
        return None
    path = latest_checkpoint(Path(cfg.out_path) / "train" / cfg.name)
    return path.resolve() if path else None
