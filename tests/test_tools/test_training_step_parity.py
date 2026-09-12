"""End-to-end parity checks for TrainModel's manual optimization loop."""

from copy import deepcopy
from math import exp
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from yolo.tools.solver import TrainModel
from yolo.utils.model_utils import EMA


_INPUTS = torch.tensor(
    [
        [1.0, -1.0],
        [0.5, 2.0],
        [-1.5, 0.25],
        [2.0, 1.0],
        [-0.5, -2.0],
        [1.25, 0.75],
        [-2.0, 1.5],
        [0.1, -0.3],
        [3.0, -0.5],
    ],
    dtype=torch.float64,
)
_TARGETS = torch.tensor([[0.4], [-1.2], [1.7], [0.3], [-0.8], [1.1], [2.2], [-0.1], [0.9]], dtype=torch.float64)
_INITIAL_WEIGHT = torch.tensor([[0.35, -0.2]], dtype=torch.float64)
_INITIAL_BIAS = torch.tensor([0.1], dtype=torch.float64)


class _PredictionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, dtype=torch.float64)
        with torch.no_grad():
            self.linear.weight.copy_(_INITIAL_WEIGHT)
            self.linear.bias.copy_(_INITIAL_BIAS)

    def forward(self, inputs: Tensor):
        prediction = self.linear(inputs)
        return {"Main": prediction, "AUX": prediction}


class _IdentityConverter:
    def __init__(self) -> None:
        self.image_sizes = []

    def __call__(self, prediction: Tensor) -> Tensor:
        return prediction

    def update(self, image_size) -> None:
        self.image_sizes.append(list(image_size))


class _RecordingSGD(SGD):
    def __init__(self, params, *, owner: "_TinyTrainModel", **kwargs) -> None:
        self.owner = owner
        self.step_attempts = []
        self.epoch_calls = []
        self.batch_calls = 0
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        self.step_attempts.append(self.owner.active_ni)
        return super().step(closure=closure)

    def next_epoch(self, batch_num: int, epoch_idx: int) -> None:
        self.epoch_calls.append((batch_num, epoch_idx))

    def next_batch(self):
        self.batch_calls += 1
        return {}


def _collate(samples):
    inputs, targets = zip(*samples)
    return len(samples), torch.stack(inputs), torch.stack(targets), None, None


def _config(*, batch_size: int, equivalent_batch_size: int, warmup_epochs: float, clip_val: float):
    return OmegaConf.create(
        {
            "image_size": [1, 1],
            "task": {
                "data": {"batch_size": batch_size, "equivalent_batch_size": equivalent_batch_size},
                "scheduler": {"warmup": {"epochs": warmup_epochs, "min_iterations": 0}},
                "gradient_clip_val": clip_val,
                "gradient_clip_algorithm": "norm",
            },
        }
    )


def _loss(aux_prediction: Tensor, main_prediction: Tensor, targets: Tensor):
    loss = F.mse_loss((aux_prediction + main_prediction) / 2, targets)
    return loss, {"Loss/MSE": loss.detach()}


class _TinyTrainModel(TrainModel):
    """Use the production training hooks without constructing YOLO or its datasets."""

    def __init__(self, cfg, inputs: Tensor, targets: Tensor, *, result_path: str | None = None) -> None:
        LightningModule.__init__(self)
        self.automatic_optimization = False
        self.cfg = cfg
        self.model = _PredictionHead()
        self.vec2box = _IdentityConverter()
        self.loss_fn = _loss
        self.train_loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=cfg.task.data.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )
        self.returned_losses = []
        self.active_ni = -1
        self.result_path = result_path

    def setup(self, stage) -> None:
        # The expensive parent setup constructs validation metrics and YOLO
        # converters. This fixture supplies the three training dependencies
        # directly while retaining the production optimization hooks.
        pass

    def val_dataloader(self):
        return None

    def training_step(self, batch, batch_idx):
        self.active_ni = self.current_epoch * self.trainer.num_training_batches + batch_idx
        output = super().training_step(batch, batch_idx)
        self.returned_losses.append(output.detach().cpu())
        return output

    def configure_optimizers(self):
        optimizer = _RecordingSGD(self.model.parameters(), owner=self, lr=0.1, momentum=0.6)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        return [optimizer], [scheduler]

    def on_train_end(self) -> None:
        if self.result_path is None or self.global_rank != 0:
            return
        optimizer = self.optimizers(use_pl_optimizer=False)
        scheduler = self.lr_schedulers()
        torch.save(
            {
                "model": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "attempts": optimizer.step_attempts,
                "scheduler": scheduler.state_dict(),
            },
            self.result_path,
        )


def _accumulation_factor(ni: int, *, batches_per_epoch: int, warmup_epochs: float, ratio: float) -> int:
    warmup_batches = max(round(warmup_epochs * batches_per_epoch), 0)
    if warmup_batches and ni < warmup_batches:
        ratio = 1 + (ratio - 1) * min(ni / warmup_batches, 1)
    return max(1, round(ratio))


def _reference_train(
    inputs: Tensor,
    targets: Tensor,
    *,
    epochs: int,
    local_batch_size: int,
    world_size: int,
    equivalent_batch_size: int,
    warmup_epochs: float,
    clip_val: float,
    distributed: bool = False,
):
    model = _PredictionHead()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.6)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    if distributed:
        rank_indices = [list(range(rank, len(inputs), world_size)) for rank in range(world_size)]
        batches_per_epoch = len(rank_indices[0]) // local_batch_size
        batches = [
            sum(
                (indices[offset : offset + local_batch_size] for indices in rank_indices),
                [],
            )
            for offset in range(0, len(rank_indices[0]), local_batch_size)
        ]
    else:
        batches_per_epoch = (len(inputs) + local_batch_size - 1) // local_batch_size
        batches = [
            list(range(offset, min(offset + local_batch_size, len(inputs))))
            for offset in range(0, len(inputs), local_batch_size)
        ]

    ratio = equivalent_batch_size / (local_batch_size * world_size)
    last_opt_step = -1
    attempts = []
    returned_losses = []
    ema_state = deepcopy(model.state_dict())
    ema_step = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        for batch_idx, indices in enumerate(batches):
            ni = epoch * batches_per_epoch + batch_idx
            prediction = model(inputs[indices])["Main"]
            loss = F.mse_loss(prediction, targets[indices])
            local_count = len(indices) // world_size if distributed else len(indices)
            returned_losses.append((loss * local_count).detach().clone())
            # DDP averages rank gradients. Scaling each local mean by
            # local_batch*world_size is therefore the global example sum.
            (loss * len(indices)).backward()
            accumulation = _accumulation_factor(
                ni,
                batches_per_epoch=batches_per_epoch,
                warmup_epochs=warmup_epochs,
                ratio=ratio,
            )
            if ni - last_opt_step < accumulation:
                continue
            clip_grad_norm_(model.parameters(), clip_val, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            attempts.append(ni)
            last_opt_step = ni
            ema_step += 1
            decay = 0.9 * (1 - exp(-ema_step / 2.0))
            ema_state = {
                key: current.detach() + (ema_state[key] - current.detach()) * decay
                for key, current in model.state_dict().items()
            }
        scheduler.step()
    return model, optimizer, scheduler, attempts, returned_losses, ema_state


def _momentum_buffers(optimizer_state):
    return [state["momentum_buffer"] for state in optimizer_state["state"].values()]


def test_train_model_enables_manual_optimization_in_its_constructor():
    cfg = OmegaConf.create(
        {
            "model": {},
            "dataset": {"class_num": 1},
            "weight": False,
            "task": {
                "task": "train",
                "data": {},
                "validation": {"task": "validation", "data": {}},
            },
        }
    )
    with (
        patch("yolo.tools.solver.create_model", return_value=_PredictionHead()),
        patch("yolo.tools.solver.create_dataloader", return_value=[]),
        patch("yolo.tools.solver.create_validation_metric", return_value=object()),
    ):
        module = TrainModel(cfg)

    assert module.automatic_optimization is False
    assert module._last_opt_step == -1


@pytest.mark.parametrize("clip_val", [0.15, 1e6], ids=["clipped", "unclipped"])
def test_training_step_matches_original_microbatch_clock_and_normalization(clip_val):
    cfg = _config(batch_size=2, equivalent_batch_size=8, warmup_epochs=1.0, clip_val=clip_val)
    module = _TinyTrainModel(cfg, _INPUTS, _TARGETS)
    ema = EMA(decay=0.9, tau=2.0)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        precision="64-true",
        callbacks=[ema],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        deterministic=True,
    )

    trainer.fit(module)
    reference = _reference_train(
        _INPUTS,
        _TARGETS,
        epochs=2,
        local_batch_size=2,
        world_size=1,
        equivalent_batch_size=8,
        warmup_epochs=1.0,
        clip_val=clip_val,
    )
    (
        expected_model,
        expected_optimizer,
        expected_scheduler,
        expected_attempts,
        expected_losses,
        expected_ema,
    ) = reference
    optimizer = module.optimizers(use_pl_optimizer=False)
    scheduler = module.lr_schedulers()

    assert module.automatic_optimization is False
    assert expected_attempts == [0, 2, 6]
    assert optimizer.step_attempts == expected_attempts
    assert optimizer.epoch_calls == [(5, 0), (5, 1)]
    assert optimizer.batch_calls == 10
    assert module.vec2box.image_sizes == [[1, 1], [1, 1]]
    assert all(not output.requires_grad for output in module.returned_losses)
    torch.testing.assert_close(torch.stack(module.returned_losses), torch.stack(expected_losses), rtol=0, atol=1e-12)
    for actual, expected in zip(module.model.parameters(), expected_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    for actual, expected in zip(
        _momentum_buffers(optimizer.state_dict()), _momentum_buffers(expected_optimizer.state_dict())
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_optimizer.param_groups[0]["lr"])
    assert scheduler.state_dict() == expected_scheduler.state_dict()
    assert ema.step == len(expected_attempts)
    assert ema.tau == 2.0
    for key, expected in expected_ema.items():
        torch.testing.assert_close(ema.ema_state_dict[key], expected, rtol=0, atol=1e-12)


def test_fixed_accumulation_discards_epoch_tail_without_resetting_iteration_clock():
    cfg = _config(batch_size=2, equivalent_batch_size=8, warmup_epochs=0.0, clip_val=0.15)
    module = _TinyTrainModel(cfg, _INPUTS, _TARGETS)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        precision="64-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        deterministic=True,
    )

    trainer.fit(module)
    expected_model, expected_optimizer, expected_scheduler, expected_attempts, *_ = _reference_train(
        _INPUTS,
        _TARGETS,
        epochs=2,
        local_batch_size=2,
        world_size=1,
        equivalent_batch_size=8,
        warmup_epochs=0.0,
        clip_val=0.15,
    )
    optimizer = module.optimizers(use_pl_optimizer=False)

    assert optimizer.step_attempts == expected_attempts == [3, 7]
    for actual, expected in zip(module.model.parameters(), expected_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    for actual, expected in zip(
        _momentum_buffers(optimizer.state_dict()), _momentum_buffers(expected_optimizer.state_dict())
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert module.lr_schedulers().state_dict() == expected_scheduler.state_dict()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="Gloo distributed backend unavailable",
)
@pytest.mark.parametrize("clip_val", [0.2, 1e6], ids=["clipped", "unclipped"])
def test_training_step_ddp_world_size_scaling_matches_global_sum(tmp_path, clip_val):
    inputs = _INPUTS[:8]
    targets = _TARGETS[:8]
    cfg = _config(batch_size=2, equivalent_batch_size=8, warmup_epochs=0.0, clip_val=clip_val)
    result_path = tmp_path / "ddp-result.pt"
    module = _TinyTrainModel(cfg, inputs, targets, result_path=str(result_path))
    trainer = Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(process_group_backend="gloo", start_method="spawn"),
        max_epochs=2,
        precision="64-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        deterministic=True,
    )

    trainer.fit(module)
    actual = torch.load(result_path, weights_only=True)
    expected_model, expected_optimizer, expected_scheduler, expected_attempts, *_ = _reference_train(
        inputs,
        targets,
        epochs=2,
        local_batch_size=2,
        world_size=2,
        equivalent_batch_size=8,
        warmup_epochs=0.0,
        clip_val=clip_val,
        distributed=True,
    )

    assert actual["attempts"] == expected_attempts == [1, 3]
    for key, expected in expected_model.state_dict().items():
        torch.testing.assert_close(actual["model"][key], expected, rtol=0, atol=1e-12)
    for actual_buffer, expected_buffer in zip(
        _momentum_buffers(actual["optimizer"]), _momentum_buffers(expected_optimizer.state_dict())
    ):
        torch.testing.assert_close(actual_buffer, expected_buffer, rtol=0, atol=1e-12)
    assert actual["optimizer"]["param_groups"][0]["lr"] == pytest.approx(expected_optimizer.param_groups[0]["lr"])
    assert actual["scheduler"] == expected_scheduler.state_dict()
