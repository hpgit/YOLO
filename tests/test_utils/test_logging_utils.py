import re
from types import SimpleNamespace

import pytest
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from yolo.utils import logging_utils
from yolo.utils.logging_utils import YOLOQuietEpochSummary, YOLORichProgressBar


class ProgressTestModel(LightningModule):
    """Exercise real Lightning callback hooks without a detector or dataset download."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        loss = self.layer(batch[0]).square().mean()
        self.log("Train/BoxLoss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.layer(batch[0])

    def on_validation_epoch_end(self):
        names = (
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
        self.log_dict({name: 0.0 for name in names}, prog_bar=True, logger=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.parametrize("with_logger", [False, True])
def test_progress_bar_sanity_training_and_validation(tmp_path, with_logger):
    progress = YOLORichProgressBar()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        callbacks=[progress],
        logger=CSVLogger(tmp_path) if with_logger else [],
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=2,
        default_root_dir=tmp_path,
    )
    loader = DataLoader(TensorDataset(torch.ones(4, 2)), batch_size=2)
    trainer.fit(ProgressTestModel(), train_dataloaders=loader, val_dataloaders=loader)

    assert trainer.global_step == 4
    assert len(progress.past_results) == 2
    assert "Train/BoxLoss_epoch" in trainer.callback_metrics


class QuietSummaryTestModel(ProgressTestModel):
    def training_step(self, batch, batch_idx):
        # Uneven batches make the epoch mean differ from both the last batch
        # and the unweighted mean of batch means.
        value = batch[0].mean() + self.current_epoch
        self.log_dict(
            {f"Loss/{name}Loss": value * scale for name, scale in (("Box", 1), ("DFL", 2), ("BCE", 3))},
            on_step=True, on_epoch=True, batch_size=len(batch[0]), logger=False,
        )
        return self.layer(batch[0]).square().mean()

    def on_validation_epoch_end(self):
        self.log_dict(
            {name: (self.current_epoch + 1) / 10 for name in YOLOQuietEpochSummary.metric_labels},
            logger=False,
        )
        self.log("map_large", -1.0, logger=False)


@pytest.mark.parametrize("validation_interval", [1, 2])
def test_quiet_summary_epoch_averages_after_validation(tmp_path, capsys, validation_interval):
    trainer = Trainer(
        accelerator="cpu", devices=1, max_epochs=2,
        callbacks=[YOLOQuietEpochSummary()], logger=False,
        enable_progress_bar=False, enable_checkpointing=False, enable_model_summary=False,
        num_sanity_val_steps=2, check_val_every_n_epoch=validation_interval,
        default_root_dir=tmp_path,
    )
    loader = DataLoader(TensorDataset(torch.tensor([[1., 1.], [3., 3.], [5., 5.]])), batch_size=2)
    trainer.fit(QuietSummaryTestModel(), train_dataloaders=loader, val_dataloaders=loader)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 // validation_interval
    for epoch, line in zip(range(validation_interval, 3, validation_interval), lines):
        assert line.startswith(f"Epoch {epoch} | ")
        for stage in ("train", "validation"):
            assert re.search(rf"{stage}=\d+\.\d{{2}}s \(\d+\.\d{{2}} it/s\)", line)
        assert f"Loss/BoxLoss={epoch + 2:.4f}" in line
        assert f"Loss/DFLLoss={(epoch + 2) * 2:.4f}" in line
        assert f"Loss/BCELoss={(epoch + 2) * 3:.4f}" in line
        assert "_step" not in line and "_epoch" not in line
        for label in YOLOQuietEpochSummary.metric_labels.values():
            expected = "N/A" if label == "AP_large" else f"{epoch * 10:.2f}%"
            assert f"{label}={expected}" in line


@pytest.mark.parametrize("quiet", [False, True])
def test_setup_selects_summary_only_when_quiet(tmp_path, monkeypatch, quiet):
    monkeypatch.setattr(logging_utils, "setup_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(logging_utils, "validate_log_directory", lambda *args: tmp_path)
    monkeypatch.setattr(logging_utils.logger, "setLevel", lambda *args: None)
    monkeypatch.setattr(logging_utils.wandb.errors.term, "_log", lambda *args, **kwargs: None)
    cfg = OmegaConf.create({
        "task": {"task": "train", "data": {}}, "name": "quiet-test",
        "quiet": quiet, "use_tensorboard": False, "use_wandb": False,
    })

    callbacks, loggers, _ = logging_utils.setup(cfg)

    assert any(isinstance(callback, YOLOQuietEpochSummary) for callback in callbacks) == quiet
    assert any(isinstance(callback, YOLORichProgressBar) for callback in callbacks) != quiet
    assert loggers == []


@pytest.mark.parametrize("is_global_zero", [False, True])
def test_quiet_summary_times_exclude_validation(monkeypatch, capsys, is_global_zero):
    times = iter([10.0, 16.0, 19.0, 21.0])
    monkeypatch.setattr(logging_utils, "perf_counter", lambda: next(times))
    trainer = SimpleNamespace(
        sanity_checking=False, state=SimpleNamespace(fn="fit"),
        current_epoch=0, callback_metrics={}, is_global_zero=is_global_zero,
    )
    summary = YOLOQuietEpochSummary()
    summary.on_train_epoch_start(trainer, None)
    for index in range(4):
        summary.on_train_batch_end(trainer, None, None, None, index)
    summary.on_validation_start(trainer, None)
    for index in range(6):
        summary.on_validation_batch_end(trainer, None, None, None, index)
    summary.on_validation_end(trainer, None)
    assert capsys.readouterr().out == ""
    summary.on_train_epoch_end(trainer, None)

    output = capsys.readouterr().out
    if is_global_zero:
        assert output == "Epoch 1 | train=8.00s (0.50 it/s) | validation=3.00s (2.00 it/s)\n"
    else:
        assert output == ""
