"""Observe probabilities consumed by workers, including Lightning resume/prefetch."""

from types import SimpleNamespace

import pytest
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.data import _update_dataloader
from torch.utils.data import Dataset, DistributedSampler

from yolo.tools.data_loader import MosaicScheduleDataLoader, create_dataloader
from yolo.tools.solver import TrainModel


class ProbabilityDataset(Dataset):
    def __init__(self):
        self.transform = SimpleNamespace(hyp={"mosaic": 0.4})

    def __len__(self):
        return 12

    def __getitem__(self, index):
        return self.transform.hyp["mosaic"]


class ScheduleModel(LightningModule):
    _mosaic_epoch = TrainModel._mosaic_epoch
    train_dataloader = TrainModel.train_dataloader

    def __init__(self, workers=0, close=1):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.seen = []
        self.train_loader = MosaicScheduleDataLoader(
            ProbabilityDataset(),
            batch_size=2,
            num_workers=workers,
            epoch_provider=self._mosaic_epoch,
            close_mosaic=close,
            mosaic_probability=0.4,
        )

    def training_step(self, batch, batch_idx):
        self.seen.append((self.current_epoch, batch.tolist()))
        return self.weight.square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class SaveAndStop(Callback):
    def __init__(self, path):
        self.path = path

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint(self.path)
        trainer.should_stop = True


def trainer(**kwargs):
    return Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        **kwargs,
    )


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("close", [1, 2])
def test_lightning_boundary_and_resume_use_new_probability_from_first_batch(tmp_path, workers, close):
    checkpoint = tmp_path / "resume.ckpt"
    first = ScheduleModel(workers, close)
    trainer(callbacks=[SaveAndStop(checkpoint)]).fit(first)
    assert len(first.seen) == 6
    assert all(epoch == 0 and values == [0.4, 0.4] for epoch, values in first.seen)

    resumed = ScheduleModel(workers, close)
    trainer().fit(resumed, ckpt_path=checkpoint)
    assert len(resumed.seen) == 12
    for epoch, values in resumed.seen:
        assert epoch in (1, 2)
        assert values == ([0.4, 0.4] if epoch < 3 - close else [0.0, 0.0])


@pytest.mark.parametrize("close,epoch,expected", [(0, 9, 0.4), (2, 7, 0.4), (2, 8, 0), (10, 0, 0), (11, 0, 0)])
def test_schedule_boundary(close, epoch, expected):
    loader = MosaicScheduleDataLoader(
        ProbabilityDataset(),
        epoch_provider=lambda: (epoch, 10),
        close_mosaic=close,
        mosaic_probability=0.4,
    )
    assert next(iter(loader)).item() == expected


def test_lightning_distributed_sampler_reconstruction_preserves_schedule():
    model = ScheduleModel()
    sampler = DistributedSampler(model.train_loader.dataset, num_replicas=2, rank=0)
    loader = _update_dataloader(model.train_loader, sampler)
    loader.epoch_provider = lambda: (2, 3)
    assert loader.close_mosaic == 1
    assert loader.mosaic_probability == 0.4
    assert len(list(loader)) == 3
    assert all(batch.tolist() == [0.0, 0.0] for batch in loader)


@pytest.mark.parametrize("value", [-1, 1.5, True, "15"])
def test_invalid_close_mosaic_fails_before_loading_data(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        create_dataloader(None, None, close_mosaic=value)


def test_schedule_rejects_unsupported_recipe_and_missing_epoch_provider():
    with pytest.raises(ValueError, match="YOLOv9 training"):
        create_dataloader(SimpleNamespace(data_augment={}), None, close_mosaic=1)
    with pytest.raises(ValueError, match="epoch_provider"):
        create_dataloader(SimpleNamespace(data_augment={"YOLOv9": {}}), None, close_mosaic=1)


def test_persistent_workers_rejected_instead_of_silently_using_stale_probability():
    with pytest.raises(ValueError, match="persistent_workers=False"):
        MosaicScheduleDataLoader(
            ProbabilityDataset(),
            num_workers=1,
            persistent_workers=True,
            epoch_provider=lambda: (0, 3),
            close_mosaic=1,
            mosaic_probability=0.4,
        )
