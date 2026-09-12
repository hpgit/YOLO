"""Run the project's TrainModel on full COCO and record completion evidence."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from hydra import compose, initialize_config_dir
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from yolo.tools.solver import TrainModel
from yolo.utils.model_utils import EMA, GradientAccumulation


class Audit(Callback):
    def __init__(self, output):
        self.output = output
        self.images = 0
        self.batches = 0
        self.validation_images = 0
        self.losses = []
        self.started = time.monotonic()

    def on_train_start(self, trainer, module):
        self.initial_weight = next(module.model.parameters()).detach().cpu().clone()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = float(outputs["loss"].detach())
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at batch {batch_idx}: {loss}")
        self.images += batch[0]
        self.batches += 1
        self.losses.append(loss)
        if self.batches % 100 == 0 or self.batches == 1:
            print(f"TRAIN {self.batches}/{trainer.num_training_batches} batches, {self.images} images, loss={loss:.5f}, elapsed={time.monotonic() - self.started:.0f}s", flush=True)

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            self.validation_images += batch[0]
            if batch_idx % 50 == 0:
                print(f"VALIDATION {self.validation_images} images", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 training and two validation batches using val2017; not a full COCO epoch")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("runs/train/coco-v9-t-1epoch"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(10, workers=True)
    torch.set_num_threads(4)
    config_path = Path(__file__).resolve().parents[1] / "yolo/config"
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "task=train", "dataset=coco", "model=v9-t", "weight=false", "use_wandb=false",
            "task.epoch=1", f"image_size=[{args.image_size},{args.image_size}]",
            f"task.data.batch_size={args.batch_size}", f"task.validation.data.batch_size={args.batch_size}",
            f"cpu_num={args.workers}", "dataset.auto_download=null",
            *(["dataset.train=val2017"] if args.smoke else []),
        ])
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    model = TrainModel(cfg)
    audit = Audit(args.output)
    checkpoint = ModelCheckpoint(dirpath=args.output / "checkpoints", filename="epoch-{epoch:02d}", save_last=True)
    ema = EMA(cfg.task.ema.decay)
    trainer = Trainer(
        accelerator="gpu", devices=1, max_epochs=1, precision="16-mixed",
        callbacks=[GradientAccumulation(cfg.task.data, cfg.task.scheduler), ema, audit, checkpoint],
        logger=CSVLogger(str(args.output), name="metrics"),
        log_every_n_steps=1 if args.smoke else 50,
        deterministic=True, enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=args.output,
        **({"limit_train_batches": 20, "limit_val_batches": 2} if args.smoke else {}),
    )
    trainer.fit(model)
    changed = not torch.equal(audit.initial_weight, next(model.model.parameters()).detach().cpu())
    assert changed, "Model weights did not change"
    assert all(torch.isfinite(p).all().item() for p in model.parameters()), "Non-finite model weights"
    assert trainer.current_epoch == 1
    if not args.smoke:
        assert audit.images == 118287, audit.images
        assert audit.validation_images == 5000, audit.validation_images
    saved = torch.load(checkpoint.last_model_path, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 0 and saved["global_step"] > 0
    result = {
        "status": "passed", "full_coco_epoch": not args.smoke,
        "model": "v9-t", "pretrained": False, "image_size": args.image_size,
        "batch_size": args.batch_size, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "train_images": audit.images,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "train_batches": audit.batches, "validation_images": audit.validation_images,
        "completed_epochs": trainer.current_epoch, "optimizer_steps": trainer.global_step,
        "optimizer_step_attempts": trainer.global_step, "successful_optimizer_steps": ema.step,
        "amp_skipped_steps": trainer.global_step - ema.step,
        "finite_losses": True, "weights_changed": changed,
        "first_loss": audit.losses[0], "last_loss": audit.losses[-1],
        "elapsed_seconds": time.monotonic() - audit.started,
        "metrics": {key: float(value) for key, value in trainer.callback_metrics.items() if value.numel() == 1},
        "checkpoint": str(Path(checkpoint.last_model_path).resolve()),
    }
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
