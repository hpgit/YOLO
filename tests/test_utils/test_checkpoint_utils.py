from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning import LightningModule, Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from yolo import lazy
from yolo.model.yolo import YOLO
from yolo.utils.checkpoint_utils import YOLOCheckpoint, latest_checkpoint, resolve_training_checkpoint
from yolo.utils import logging_utils
from yolo.utils.logging_utils import setup, validate_log_directory
from yolo.utils.model_utils import EMA


def config(tmp_path, **kwargs):
    return OmegaConf.create({
        "task": {"task": "train", "data": {}}, "name": "run", "weight": True,
        "out_path": str(tmp_path), "exist_ok": False, "quiet": True, "lucky_number": 10,
        "use_wandb": False, "use_tensorboard": False, "device": 1, **kwargs,
    })


def snapshot(path, epoch, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "global_step": step}, path)
    return path.resolve()


def test_latest_checkpoint_uses_numeric_saved_progress_and_scopes_run(tmp_path):
    run = tmp_path / "train" / "run"
    snapshot(run / "checkpoints" / "epoch=9-step=999.ckpt", 9, 999)
    snapshot(run / "logs/version_0/checkpoints/epoch0010-step00000002.ckpt", 10, 2)
    expected = snapshot(run / "logs/version_1/checkpoints/last.ckpt", 10, 12)
    snapshot(tmp_path / "train" / "run1/checkpoints/last.ckpt", 999, 999)
    (run / "broken.ckpt").write_bytes(b"incomplete save")
    # Weight-only files never participate in resumption.
    torch.save({"epoch": 999, "global_step": 999}, run / "best.pt")
    assert latest_checkpoint(run) == expected
    assert resolve_training_checkpoint(config(tmp_path)) == expected
    assert latest_checkpoint(tmp_path / "missing") is None


@pytest.mark.parametrize("weight", [False, None, "best.pt", "pretrained.pt"])
def test_explicit_weight_overrides_named_resume(tmp_path, weight):
    snapshot(tmp_path / "train/run/checkpoints/last.ckpt", 8, 90)
    assert resolve_training_checkpoint(config(tmp_path, weight=weight)) is None


def test_explicit_checkpoint_and_boolean_overrides(tmp_path):
    snapshot(tmp_path / "train/run/checkpoints/last.ckpt", 8, 90)
    explicit = snapshot(tmp_path / "different.ckpt", 1, 2)
    assert resolve_training_checkpoint(config(tmp_path, weight=str(explicit))) == explicit
    assert resolve_training_checkpoint(config(tmp_path), weight_explicit=True) is None
    with pytest.raises(FileNotFoundError):
        resolve_training_checkpoint(config(tmp_path, weight=str(tmp_path / "missing.ckpt")))
    assert resolve_training_checkpoint(config(tmp_path, name=None)) is None
    assert resolve_training_checkpoint(config(tmp_path, task={"task": "validation"})) is None


@pytest.mark.parametrize("quiet", [False, True])
def test_setup_checkpoint_directory_and_resuming_existing_run(tmp_path, quiet):
    cfg = config(tmp_path, quiet=quiet)
    directory = tmp_path / "train/run"
    directory.mkdir(parents=True)
    callbacks, _, save_path = setup(cfg, resume=True)
    checkpoint = next(callback for callback in callbacks if isinstance(callback, YOLOCheckpoint))
    assert save_path == directory
    assert Path(checkpoint.dirpath) == directory / "checkpoints"
    assert Path(checkpoint.format_checkpoint_name({"epoch": 3, "step": 42})).name == "epoch0003-step00000042.ckpt"
    assert validate_log_directory(cfg, cfg.name) == tmp_path / "train/run1"


def test_nonzero_rank_defers_checkpoint_directory_to_lightning(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "validate_log_directory", lambda *args, **kwargs: None)
    callbacks, _, _ = setup(config(tmp_path), resume=True)
    checkpoint = next(callback for callback in callbacks if isinstance(callback, YOLOCheckpoint))
    assert checkpoint.dirpath is None


class SmallDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 1))

    def forward(self, inputs):
        return self.model(inputs)


class CheckpointModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SmallDetector()
        self.ema = self.model
        self.restored = None

    def training_step(self, batch, batch_idx):
        return self.model(batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):
        return self.ema(batch[0])

    def on_validation_epoch_end(self):
        # Epoch 0 wins, then lower, equal, invalid, and finally a new best score.
        score = [0.5, 0.3, 0.5, float("nan"), 0.8][self.current_epoch]
        self.log("map", score, logger=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return [optimizer], [scheduler]

    def on_train_start(self):
        self.restored = {
            "step": self.global_step,
            "optimizer": deepcopy(self.trainer.optimizers[0].state_dict()),
            "scheduler": deepcopy(self.lr_schedulers().state_dict()),
        }


def train(directory, epochs, checkpoint_path=None, ema=True):
    checkpoint = YOLOCheckpoint(directory)
    callbacks = [EMA()] if ema else []
    trainer = Trainer(
        accelerator="cpu", devices=1, max_epochs=epochs, callbacks=[*callbacks, checkpoint],
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=directory, num_sanity_val_steps=1,
    )
    model = CheckpointModel()
    loader = DataLoader(TensorDataset(torch.ones(4, 2)), batch_size=2)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader, ckpt_path=checkpoint_path)
    return trainer, model, checkpoint


@pytest.mark.parametrize("ema", [False, True])
def test_best_weights_and_full_resume_preserve_training_state(tmp_path, ema):
    trainer, model, callback = train(tmp_path, 1, ema=ema)
    path = latest_checkpoint(tmp_path)
    assert path.name == "epoch0000-step00000002.ckpt"
    state = torch.load(path, weights_only=False)
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    for key, value in model.ema.model.state_dict().items():
        torch.testing.assert_close(best[key], value, rtol=0, atol=0)
    # Exercise the repository's actual weight-only loader contract.
    target = SmallDetector()
    YOLO.save_load_weights(target, tmp_path / "best.pt")
    for key, value in target.model.state_dict().items():
        torch.testing.assert_close(best[key], value, rtol=0, atol=0)
    initial_bytes = (tmp_path / "best.pt").read_bytes()

    resumed, resumed_model, resumed_callback = train(tmp_path, 4, path, ema=ema)
    assert resumed.global_step == 8
    assert resumed_model.restored["step"] == 2
    assert resumed_model.restored["scheduler"] == state["lr_schedulers"][0]
    original_optimizer = state["optimizer_states"][0]
    restored_optimizer = resumed_model.restored["optimizer"]
    assert restored_optimizer["param_groups"] == original_optimizer["param_groups"]
    for index, values in original_optimizer["state"].items():
        torch.testing.assert_close(restored_optimizer["state"][index]["momentum_buffer"], values["momentum_buffer"])
    assert resumed_callback.best_map == 0.5
    assert (tmp_path / "best.pt").read_bytes() == initial_bytes
    assert latest_checkpoint(tmp_path).name == "epoch0003-step00000008.ckpt"
    if ema:
        ema_callback = next(item for item in resumed.callbacks if isinstance(item, EMA))
        assert ema_callback.step == 8

    final, final_model, final_callback = train(tmp_path, 5, latest_checkpoint(tmp_path), ema=ema)
    assert final.global_step == 10
    assert final_callback.best_map == pytest.approx(0.8)
    assert (tmp_path / "best.pt").read_bytes() != initial_bytes
    for key, value in final_model.ema.model.state_dict().items():
        torch.testing.assert_close(torch.load(tmp_path / "best.pt", weights_only=True)[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("weight", [True, False, "best.pt", "explicit.ckpt"])
@pytest.mark.parametrize("weight_override", [False, True])
def test_training_entrypoint_passes_full_checkpoint_and_avoids_pretrained_load(
    tmp_path, monkeypatch, weight, weight_override
):
    auto = snapshot(tmp_path / "train/run/checkpoints/last.ckpt", 8, 90)
    explicit = snapshot(tmp_path / "explicit.ckpt", 1, 2)
    cfg = config(tmp_path, weight=str(explicit) if weight == "explicit.ckpt" else weight)
    calls = {}

    def create_model(model_cfg):
        calls["weight"] = model_cfg.weight
        return "model"

    def fit(model, ckpt_path):
        calls["checkpoint"] = ckpt_path

    monkeypatch.setattr(lazy.HydraConfig, "initialized", lambda: True)
    overrides = [f"weight={cfg.weight}"] if weight_override else ["name=run"]
    monkeypatch.setattr(lazy.HydraConfig, "get", lambda: SimpleNamespace(overrides=SimpleNamespace(task=overrides)))
    monkeypatch.setattr(lazy, "setup", lambda cfg, resume: ([], [], tmp_path))
    monkeypatch.setattr(lazy, "Trainer", lambda **kwargs: SimpleNamespace(fit=fit))
    monkeypatch.setattr(lazy, "TrainModel", create_model)
    lazy.main.__wrapped__(cfg)
    expected = auto if weight is True and not weight_override else explicit if weight == "explicit.ckpt" else None
    assert calls["checkpoint"] == expected
    assert calls["weight"] == (False if expected else cfg.weight)
    assert cfg.weight == (str(explicit) if weight == "explicit.ckpt" else weight)
