"""Exercise dual assignment through the real trainer, EMA, checkpoints and validation."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning import Trainer
from lightning.pytorch.plugins import MixedPrecision
from PIL import Image, ImageDraw

from yolo.model.yolo import create_model
from yolo.tools.solver import TrainModel
from yolo.utils.checkpoint_utils import YOLOCheckpoint
from yolo.utils.model_utils import EMA


def _config(tmp_path):
    generator = torch.Generator().manual_seed(41)
    for index in range(2):
        image = tmp_path / "images" / "val" / f"{index}.png"
        label = tmp_path / "labels" / "val" / f"{index}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        # Distinct textured inputs avoid pathological tiny-map BatchNorm
        # gradients caused by a batch of identical constant backgrounds.
        pixels = torch.randint(0, 80, (64, 64, 3), generator=generator, dtype=torch.uint8).numpy()
        canvas = Image.fromarray(pixels)
        if index == 0:
            ImageDraw.Draw(canvas).rectangle((16, 16, 48, 48), fill=(180, 90, 40))
        canvas.save(image)
        label.write_text("0 0.5 0.5 0.5 0.5\n" if index == 0 else "")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "yolo/config"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=train",
                "model=v9-t-nms-free",
                "dataset=mock",
                "weight=false",
                "dataset.auto_download=null",
                f"dataset.path={tmp_path}",
                "dataset.train=val",
                "dataset.class_num=1",
                "dataset.class_list=[object]",
                "image_size=[64,64]",
                "cpu_num=0",
                "task.data.batch_size=2",
                "task.data.equivalent_batch_size=2",
                "task.validation.data.batch_size=2",
                "task.epoch=2",
                "task.scheduler.warmup.epochs=0",
                "task.scheduler.warmup.min_iterations=0",
                "use_wandb=false",
            ],
        )
    cfg.task.data.data_augment = {}
    return cfg


@pytest.mark.parametrize(
    "accelerator",
    ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
def test_fit_ema_checkpoint_resume_without_nms(tmp_path, accelerator):
    cfg = _config(tmp_path)
    torch.manual_seed(20)
    model = TrainModel(cfg)
    initial = {key: value.detach().clone() for key, value in model.model.named_parameters()}
    checkpoint = YOLOCheckpoint(tmp_path / "checkpoints")
    ema = EMA(cfg.task.ema.decay)
    options = dict(
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else "32-true",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        num_sanity_val_steps=1,
    )
    if accelerator == "gpu":
        # A two-step smoke cannot wait for GradScaler's default 65536 scale to
        # back off through expected warmup overflows on random YOLO weights.
        options["plugins"] = [MixedPrecision("16-mixed", "cuda", scaler=torch.amp.GradScaler("cuda", init_scale=128))]
        options.pop("precision")
    trainer = Trainer(max_epochs=1, callbacks=[ema, checkpoint], **options)
    with patch("yolo.utils.bounding_box_utils.batched_nms", side_effect=AssertionError("NMS was called")):
        trainer.fit(model)
    assert model.post_process.nms_free
    assert trainer.global_step == 1
    assert ema.step == 1
    for group in (".heads.", ".one2one_heads."):
        assert any(
            not torch.equal(initial[key], value.detach().cpu())
            for key, value in model.model.named_parameters()
            if group in key and key.endswith("weight") and value.requires_grad
        )
    loss_metrics = {key: value for key, value in trainer.callback_metrics.items() if key.startswith("Loss/")}
    assert loss_metrics and all(torch.isfinite(value) for value in loss_metrics.values())
    saved = torch.load(checkpoint.best_model_path, weights_only=False, map_location="cpu")
    assert saved["nms_free"] is True
    assert any(".one2one_heads." in key for key in saved["state_dict"])
    best = tmp_path / "checkpoints" / "best.pt"
    restored = create_model(deepcopy(cfg.model), weight_path=best, class_num=1)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, ema.ema_state_dict[key].cpu(), rtol=0, atol=0)

    resumed_model = TrainModel(cfg)
    resumed_ema = EMA(cfg.task.ema.decay)
    resumed = Trainer(max_epochs=2, callbacks=[resumed_ema, YOLOCheckpoint(tmp_path / "checkpoints")], **options)
    with patch("yolo.utils.bounding_box_utils.batched_nms", side_effect=AssertionError("NMS was called")):
        resumed.fit(resumed_model, ckpt_path=checkpoint.best_model_path)
    assert resumed.global_step == 2
    assert resumed_ema.step == 2
    assert resumed_model.post_process.nms_free
    assert torch.isfinite(resumed.callback_metrics["map"])


def test_resume_rejects_wrong_assignment_mode(tmp_path):
    model = TrainModel(_config(tmp_path))
    with pytest.raises(ValueError, match="NMS-free structure"):
        model.on_load_checkpoint({"nms_free": False})


def test_default_and_override_config_contract():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "yolo/config"), version_base=None):
        baseline = compose(config_name="config", overrides=["task=train", "model=v9-t"])
        enabled = compose(config_name="config", overrides=["task=train", "model=v9-sr-t", "model.nms_free=true"])
    assert not baseline.model.nms_free
    assert enabled.model.nms_free
    assert enabled.task.loss.one2one == 1.0
