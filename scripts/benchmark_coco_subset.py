"""Time one complete subset epoch, or validate fixed weights, on one GPU.

--source-root selects a saved source tree for an unchanged baseline. Dataset
construction and sanity validation are outside measured train/validation times.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-root", type=Path, default=ROOT)
parser.add_argument("--dataset", type=Path, default=ROOT / "data/coco-1over20")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--validation-only", action="store_true")
parser.add_argument("--eval-state", type=Path)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--image-size", type=int, default=320)
parser.add_argument("--seed", type=int, default=10)
args = parser.parse_args()
sys.path.insert(0, str(args.source_root.resolve()))
source_hashes = {str(p.relative_to(args.source_root.resolve())): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((args.source_root.resolve() / "yolo").rglob("*.py"))}

import torch
from hydra import compose, initialize_config_dir
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
import yolo
from yolo.tools.solver import TrainModel, ValidateModel
from yolo.utils.model_utils import EMA, GradientAccumulation


class Timing(Callback):
    def __init__(self, fixed_state=None):
        self.fixed_state = fixed_state
        self.setup_modified_state_keys = []
        self.train_images = self.val_images = 0
        self.losses = []
        self.predictions = []
        self.train_seconds = self.val_seconds = 0.0

    def on_train_epoch_start(self, trainer, module):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.train_images += batch[0]
        self.losses.append(outputs["loss"].detach())
        if batch_idx % 50 == 0:
            print(f"TRAIN batch={batch_idx} images={self.train_images}", flush=True)

    def on_validation_start(self, trainer, module):
        if self.fixed_state is not None:
            # The legacy auto-stride probe runs in train mode during setup and
            # changes BN buffers. Restore the exact checkpoint AFTER that probe.
            self.setup_modified_state_keys = [key for key, value in module.model.state_dict().items()
                                              if not torch.equal(value.cpu(), self.fixed_state[key])]
            module.model.load_state_dict(self.fixed_state)
            assert all(torch.equal(value.cpu(), self.fixed_state[key]) for key, value in module.model.state_dict().items())
        torch.cuda.synchronize()
        if not trainer.sanity_checking:
            if self.train_images:
                self.train_seconds = time.perf_counter() - self.start
                self.train_peak = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            self.val_start = time.perf_counter()

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            self.val_images += batch[0]
            # Identical capture overhead in both variants; enables fixed-weight equivalence checks.
            self.predictions.extend(p.detach().cpu() for p in outputs[0])

    def on_validation_end(self, trainer, module):
        if not trainer.sanity_checking:
            torch.cuda.synchronize()
            self.val_seconds = time.perf_counter() - self.val_start
            self.val_peak = torch.cuda.max_memory_allocated()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for comparable GPU performance measurements")
    if args.validation_only and args.eval_state is None:
        raise ValueError("--validation-only requires --eval-state")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise ValueError("Use a new output directory to preserve previous benchmark evidence")
    seed_everything(args.seed, workers=True)
    torch.set_num_threads(4)
    with initialize_config_dir(config_dir=str(args.source_root.resolve() / "yolo/config"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "task=train", "dataset=coco", "model=v9-t", "weight=false", "use_wandb=false",
            "task.epoch=1", "dataset.auto_download=null", f"dataset.path={args.dataset.resolve()}",
            f"image_size=[{args.image_size},{args.image_size}]", f"cpu_num={args.workers}",
            f"task.data.batch_size={args.batch_size}", f"task.validation.data.batch_size={args.batch_size}",
        ])
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    module = ValidateModel(cfg) if args.validation_only else TrainModel(cfg)
    fixed_state = torch.load(args.eval_state, map_location="cpu", weights_only=True) if args.eval_state else None
    if fixed_state is not None:
        module.model.load_state_dict(fixed_state)
    eval_state_hash = hashlib.sha256(args.eval_state.read_bytes()).hexdigest() if args.eval_state else None
    cache_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.dataset.glob("*.pache"))}
    initial = next(module.model.parameters()).detach().clone()
    timing = Timing(fixed_state if args.validation_only else None)
    callbacks = [timing] if args.validation_only else [GradientAccumulation(cfg.task.data, cfg.task.scheduler), EMA(cfg.task.ema.decay), timing]
    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=1, precision="16-mixed",
                      callbacks=callbacks, logger=CSVLogger(str(args.output), name="metrics"),
                      **({"gradient_clip_val": 10, "gradient_clip_algorithm": "norm"}
                         if module.automatic_optimization else {}),
                      log_every_n_steps=50,
                      deterministic=True, enable_progress_bar=False, enable_model_summary=False,
                      enable_checkpointing=False, default_root_dir=args.output)
    start = time.perf_counter()
    if args.validation_only:
        trainer.validate(module)
    else:
        trainer.fit(module)
    elapsed = time.perf_counter() - start
    assert timing.val_images == len(module.val_loader.dataset)
    if not args.validation_only:
        assert timing.train_images == len(module.train_loader.dataset)
        assert not torch.equal(initial, next(module.model.parameters()).detach().cpu())
        assert torch.isfinite(torch.stack(timing.losses)).all().item()
    assert all(torch.isfinite(p).all().item() for p in module.parameters())
    torch.save(module.ema.state_dict(), args.output / "ema.pt")
    torch.save(timing.predictions, args.output / "predictions.pt")
    manifest = (args.dataset / "manifest.json").read_bytes()
    assert source_hashes == {str(p.relative_to(args.source_root.resolve())): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((args.source_root.resolve() / "yolo").rglob("*.py"))}, "Source changed during benchmark"
    assert cache_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.dataset.glob("*.pache"))}, "Dataset cache changed during benchmark"
    result = {
        "source": yolo.__file__, "source_sha256": source_hashes, "dataset_cache_sha256": cache_hashes,
        "dataset_manifest_sha256": hashlib.sha256(manifest).hexdigest(), "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "seed": args.seed, "image_size": args.image_size, "batch_size": args.batch_size,
        "workers": args.workers, "validation_only": args.validation_only,
        "eval_state": str(args.eval_state.resolve()) if args.eval_state else None, "eval_state_sha256": eval_state_hash,
        "setup_modified_state_keys_restored": timing.setup_modified_state_keys,
        "train_images": timing.train_images, "val_images": timing.val_images,
        "train_seconds": timing.train_seconds, "validation_seconds": timing.val_seconds,
        "fit_or_validate_seconds": elapsed, "train_images_per_second": timing.train_images / timing.train_seconds if timing.train_images else None,
        "validation_images_per_second": timing.val_images / timing.val_seconds,
        "train_peak_allocated_bytes": getattr(timing, "train_peak", None), "val_peak_allocated_bytes": timing.val_peak,
        "losses": [float(x) for x in timing.losses], "finite_weights": True,
        "metrics": {k: float(v) for k, v in trainer.callback_metrics.items() if v.numel() == 1},
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_sha256", "losses", "metrics")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
