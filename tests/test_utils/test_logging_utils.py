import pytest
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from yolo.utils.logging_utils import YOLORichProgressBar


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
