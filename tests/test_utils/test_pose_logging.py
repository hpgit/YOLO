import io
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning import LightningModule, Trainer
from rich.console import Console
from torch.utils.data import DataLoader, TensorDataset

from yolo.utils import logging_utils
from yolo.utils.coco_eval import METRIC_NAMES
from yolo.utils.logging_utils import (
    ImageLogger,
    YOLOQuietEpochSummary,
    YOLORichProgressBar,
    log_bbox,
)
from yolo.utils.pose_eval import POSE_METRIC_NAMES


def pose_metrics():
    # Deliberately reverse insertion order and include non-summary metrics.
    result = {name: torch.tensor((index + 1) / 100) for index, name in reversed(list(enumerate(POSE_METRIC_NAMES)))}
    result["map_large"] = torch.tensor(-1.0)
    result["map_step"] = torch.tensor(0.99)
    return result


def test_rich_callback_uses_named_pose_metrics_and_ar20(monkeypatch):
    progress = YOLORichProgressBar()
    progress.progress = SimpleNamespace()
    progress.max_result = 0
    progress.past_results = deque(maxlen=5)
    monkeypatch.setattr(progress, "get_metrics", lambda *_: pose_metrics())
    trainer = SimpleNamespace(state=SimpleNamespace(fn="validate"), current_epoch=2)
    progress.on_validation_end(trainer, None)
    assert len(progress.past_results) == 1
    _, summary = progress.past_results[0]
    assert summary[2] == pytest.approx(1.0)
    assert summary[5] == pytest.approx(6.0)
    output = io.StringIO()
    Console(file=output, width=150, color_system=None).print(progress.progress.table)
    text = output.getvalue()
    assert "COCO Pose (OKS, maxDets=20)" in text
    assert "Pose AR20" in text and "N/A" in text
    assert "maxDets 100" not in text and "AR100" not in text


def test_detection_progress_uses_explicit_coco_order(monkeypatch):
    progress = YOLORichProgressBar()
    progress.progress = SimpleNamespace()
    progress.max_result = 0
    progress.past_results = deque(maxlen=5)
    metrics = {name: (index + 1) / 100 for index, name in reversed(list(enumerate(METRIC_NAMES)))}
    monkeypatch.setattr(progress, "get_metrics", lambda *_: metrics)
    progress.on_validation_end(SimpleNamespace(state=SimpleNamespace(fn="validate"), current_epoch=0), None)
    np.testing.assert_allclose(progress.max_result, np.arange(1, 13))


def test_pose_quiet_summary_includes_pose_loss_and_ar20(tmp_path, capsys):
    metrics = pose_metrics()
    metrics.update({"Loss/PoseDFLoss_epoch": 2.0, "Loss/PoseVisibilityLoss_epoch": 0.5})
    trainer = SimpleNamespace(
        callback_metrics=metrics, is_global_zero=True, state=SimpleNamespace(fn="fit"), current_epoch=0
    )
    path = tmp_path / "result.log"
    YOLOQuietEpochSummary(path)._print_summary(trainer)
    output = capsys.readouterr().out
    assert "PoseAP=1.00%" in output and "PoseAR20=6.00%" in output
    assert "PoseAP_large=N/A" in output
    assert "Loss/PoseDFLoss=2.0000" in output
    assert "Loss/PoseVisibilityLoss=0.5000" in output
    assert "AR100" not in output and "map_step" not in output
    assert path.read_text() == output


def test_image_logger_slices_pose_columns_before_bbox_logging(monkeypatch):
    class FakeWandbLogger:
        def __init__(self):
            self.entries = []

        def log_image(self, name, images, **kwargs):
            self.entries.append((name, kwargs))

    monkeypatch.setattr(logging_utils, "WandbLogger", FakeWandbLogger)
    recorder = FakeWandbLogger()
    trainer = SimpleNamespace(current_epoch=3, loggers=[recorder])
    targets = torch.tensor([[[0.0, 10.0, 20.0, 30.0, 40.0] + [999.0, 998.0, 2.0] * 17]])
    prediction = torch.tensor([[0.0, 10.0, 20.0, 30.0, 40.0, 0.75] + [999.0, 998.0, 0.9] * 17])
    batch = (1, torch.zeros(1, 3, 64, 64), targets, None, ["image.jpg"])
    ImageLogger().on_validation_batch_end(trainer, None, ([prediction], None), batch, 0)
    ground_truth = recorder.entries[1][1]["boxes"][0]["predictions"]["box_data"][0]
    predicted = recorder.entries[2][1]["boxes"][0]["predictions"]["box_data"][0]
    assert "scores" not in ground_truth
    assert predicted["scores"]["confidence"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="expects"):
        log_bbox(prediction)


class PoseProgressModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        loss = self.layer(batch[0]).square().mean()
        self.log("Loss/PoseDFLoss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.layer(batch[0])

    def on_validation_epoch_end(self):
        self.log_dict(
            {name: value for name, value in pose_metrics().items() if name in POSE_METRIC_NAMES},
            prog_bar=True,
            logger=False,
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_real_lightning_pose_sanity_training_and_validation(tmp_path):
    progress = YOLORichProgressBar()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        callbacks=[progress],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
        default_root_dir=tmp_path,
    )
    loader = DataLoader(TensorDataset(torch.ones(4, 2)), batch_size=2)
    trainer.fit(PoseProgressModel(), train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.global_step == 4 and len(progress.past_results) == 2
    assert "Loss/PoseDFLoss_epoch" in trainer.callback_metrics
    assert progress.max_result.shape == (10,)
