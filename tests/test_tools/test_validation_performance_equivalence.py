"""Regression checks for skipping unused validation work."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torchmetrics.detection import MeanAveragePrecision

from yolo.model.yolo import create_model
from yolo.tools.solver import ValidateModel


@pytest.mark.parametrize("name", ["v9-t", "v9-s", "v9-m", "v9-c", "v7"])
def test_shortcut_preserves_main_and_skips_auxiliary(name):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "yolo/config/model" / f"{name}.yaml")
    model = create_model(cfg, weight_path=False).eval()
    images = torch.rand(1, 3, 64, 64)
    called = []
    hooks = [layer.register_forward_hook(lambda *unused: called.append(True))
             for index, layer in enumerate(model.model, start=1) if index > model.layer_index["Main"]]
    with torch.inference_mode():
        full = model(images)["Main"]
        called.clear()
        shortcut = model(images, shortcut="Main")["Main"]
    assert not called
    for hook in hooks:
        hook.remove()

    def compare(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert len(left) == len(right)
            for a, b in zip(left, right):
                compare(a, b)
    compare(full, shortcut)


def test_update_only_has_same_epoch_metrics():
    before = MeanAveragePrecision(backend="faster_coco_eval")
    after = MeanAveragePrecision(backend="faster_coco_eval")
    for label in (0, 1):
        targets = [{"boxes": torch.tensor([[1., 2., 5., 8.]]), "labels": torch.tensor([label])}]
        preds = [{**targets[0], "scores": torch.tensor([0.8])}]
        before(preds, targets)
        after.update(preds, targets)
    for key, value in before.compute().items():
        torch.testing.assert_close(value, after.compute()[key], rtol=0, atol=0)


def test_validation_step_updates_without_computing():
    module = SimpleNamespace(ema=Mock(), metric=Mock(),
                             post_process=Mock(return_value=[torch.tensor([[0., 1., 2., 5., 8., .8]])]))
    batch = (1, torch.rand(1, 3, 64, 64), torch.tensor([[[0., 1., 2., 5., 8.]]]), None, None)
    predictions, batch_map = ValidateModel.validation_step(module, batch, 0)
    assert len(predictions) == 1 and batch_map is None
    module.metric.update.assert_called_once()
    module.metric.assert_not_called()
    module.metric.compute.assert_not_called()
    assert module.ema.call_args.kwargs == {"shortcut": "Main"}


@pytest.mark.parametrize("world_size,expected", [(1, True), (2, False)])
def test_metric_cpu_storage_is_disabled_for_distributed_validation(world_size, expected):
    trainer = SimpleNamespace(world_size=world_size)
    module = SimpleNamespace(_trainer=trainer, trainer=trainer, metric=SimpleNamespace(compute_on_cpu=False),
        cfg=SimpleNamespace(model=SimpleNamespace(name="v9-t", anchor=None), image_size=[64, 64]),
        model=None, device=torch.device("cpu"), validation_cfg=SimpleNamespace(nms=None))
    with patch("yolo.tools.solver.create_converter"), patch("yolo.tools.solver.PostProcess"):
        ValidateModel.setup(module, "validate")
    assert module.metric.compute_on_cpu is expected
