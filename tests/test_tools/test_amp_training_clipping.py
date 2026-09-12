"""Exercise production clipping through Lightning's real CUDA AMP step."""

from math import exp

import pytest
import torch
from lightning import LightningModule, Trainer
from omegaconf import OmegaConf
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from yolo.tools.solver import TrainModel
from yolo.utils.model_utils import EMA


class _ScalarHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))

    def forward(self, inputs):
        # Elementwise FP32 math avoids incidental FP16 overflow from AMP's
        # initial scale. The middle batch deliberately supplies infinity.
        prediction = self.weight * inputs
        return {"Main": prediction, "AUX": prediction}


class _IdentityConverter:
    def __call__(self, prediction):
        return prediction

    def update(self, image_size):
        pass


class _ConstantSGD(SGD):
    def next_epoch(self, batch_num, epoch_idx):
        pass

    def next_batch(self):
        return {}


def _collate(samples):
    inputs = torch.stack([sample[0] for sample in samples])
    return len(samples), inputs, torch.zeros_like(inputs), None, None


class _AmpTrainModel(TrainModel):
    """Keep TrainModel's backward, step, clipping and scheduler hooks intact."""

    def __init__(self):
        LightningModule.__init__(self)
        self.automatic_optimization = False
        self._last_opt_step = -1
        self.cfg = OmegaConf.create(
            {
                "image_size": [1, 1],
                "task": {
                    "data": {"batch_size": 1, "equivalent_batch_size": 1},
                    "scheduler": {"warmup": {"epochs": 0, "min_iterations": 100}},
                    "gradient_clip_val": 2.0,
                    "gradient_clip_algorithm": "norm",
                },
            }
        )
        self.model = _ScalarHead()
        self.vec2box = _IdentityConverter()
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor([[4.0], [float("inf")], [4.0]])),
            batch_size=1,
            num_workers=0,
            collate_fn=_collate,
        )
        self.gradients_before_clip = []
        self.gradients_after_clip = []
        self.batch_states = []

    def setup(self, stage):
        pass

    def val_dataloader(self):
        return None

    @staticmethod
    def loss_fn(aux, main, targets):
        loss = main.mean()
        return loss, {"Loss/linear": loss.detach()}

    def configure_optimizers(self):
        optimizer = _ConstantSGD(self.model.parameters(), lr=0.25, momentum=0.5)
        return [optimizer], [StepLR(optimizer, step_size=1, gamma=1.0)]

    def on_before_optimizer_step(self, optimizer):
        self.gradients_before_clip.append(self.model.weight.grad.detach().cpu().clone())
        super().on_before_optimizer_step(optimizer)
        self.gradients_after_clip.append(self.model.weight.grad.detach().cpu().clone())

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        optimizer = self.optimizers(use_pl_optimizer=False)
        ema = next(callback for callback in self.trainer.callbacks if isinstance(callback, EMA))
        self.batch_states.append(
            {
                "weight": self.model.weight.detach().cpu().clone(),
                "momentum": optimizer.state[self.model.weight]["momentum_buffer"].detach().cpu().clone(),
                "ema_weight": ema.ema_state_dict["weight"].detach().cpu().clone(),
                "ema_step": ema.step,
                "scale": self.trainer.precision_plugin.scaler.get_scale(),
            }
        )
        return result


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_lightning_unscales_before_production_clipping_and_skips_overflow():
    module = _AmpTrainModel()
    ema = EMA(decay=0.9, tau=2.0)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=1,
        callbacks=[ema],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    trainer.fit(module)

    assert trainer.precision_plugin.scaler is not None
    first, overflow, last = module.batch_states
    assert first["scale"] > 1
    assert overflow["scale"] == first["scale"] / 2
    assert last["scale"] == overflow["scale"]

    # d(weight * 4)/d(weight) = 4, independent of either GradScaler scale.
    # Seeing 4 at this hook directly proves unscale occurs before clipping.
    for index in (0, 2):
        torch.testing.assert_close(module.gradients_before_clip[index], torch.tensor([4.0]), rtol=0, atol=0)
    assert torch.isinf(module.gradients_before_clip[1]).all()
    clipped = 4.0 * (2.0 / (4.0 + 1e-6))
    for index in (0, 2):
        torch.testing.assert_close(module.gradients_after_clip[index], torch.tensor([clipped]))

    # Two successful SGD updates, separated by one overflow: v1=g, v2=.5g+g.
    first_weight = 2.0 - 0.25 * clipped
    final_weight = first_weight - 0.25 * (1.5 * clipped)
    torch.testing.assert_close(first["weight"], torch.tensor([first_weight]))
    torch.testing.assert_close(first["momentum"], torch.tensor([clipped]))
    torch.testing.assert_close(last["weight"], torch.tensor([final_weight]))
    torch.testing.assert_close(last["momentum"], torch.tensor([1.5 * clipped]))

    assert [state["ema_step"] for state in module.batch_states] == [1, 1, 2]
    for key in ("weight", "momentum", "ema_weight"):
        torch.testing.assert_close(overflow[key], first[key], rtol=0, atol=0)
    first_decay = 0.9 * (1 - exp(-1 / 2.0))
    first_ema = first_weight + (2.0 - first_weight) * first_decay
    second_decay = 0.9 * (1 - exp(-2 / 2.0))
    final_ema = final_weight + (first_ema - final_weight) * second_decay
    torch.testing.assert_close(first["ema_weight"], torch.tensor([first_ema]))
    torch.testing.assert_close(last["ema_weight"], torch.tensor([final_ema]))
    assert module._last_opt_step == 2
