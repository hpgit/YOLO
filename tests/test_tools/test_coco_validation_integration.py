"""Check official JSON evaluation wiring, including training's validation config."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf
from torchmetrics.detection import MeanAveragePrecision

from yolo.tools.solver import ValidateModel, create_validation_metric
from yolo.utils.coco_eval import CocoJsonEvaluator


@pytest.fixture
def configs(tmp_path):
    annotation = tmp_path / "annotations" / "instances_val.json"
    annotation.parent.mkdir()
    annotation.write_text(json.dumps({"info": {}, "images": [{"id": 42, "file_name": "sample.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 7, "name": "object"}],
        "annotations": [{"id": 1, "image_id": 42, "category_id": 7, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]}))
    return OmegaConf.create({"task": "validation", "evaluator": "auto", "annotation_path": None,
                             "data": {"data_augment": {}}}), OmegaConf.create({"path": str(tmp_path), "validation": "val", "class_num": 1})


def test_auto_uses_json_and_explicit_legacy_is_available(configs):
    validation, dataset = configs
    assert isinstance(create_validation_metric(validation, dataset), CocoJsonEvaluator)
    validation.evaluator = "torchmetrics"
    assert isinstance(create_validation_metric(validation, dataset), MeanAveragePrecision)


def test_auto_falls_back_only_when_no_json_was_requested(configs):
    validation, dataset = configs
    dataset.validation = "txt_split"
    assert isinstance(create_validation_metric(validation, dataset), MeanAveragePrecision)
    validation.evaluator = "coco"
    with pytest.raises(FileNotFoundError):
        create_validation_metric(validation, dataset)
    validation.evaluator = "auto"
    validation.annotation_path = "missing.json"
    with pytest.raises(FileNotFoundError):
        create_validation_metric(validation, dataset)


def test_split_txt_precedence_is_explicit_when_json_also_exists(configs):
    from pathlib import Path

    validation, dataset = configs
    (Path(dataset.path) / "val.txt").write_text("images/val/sample.jpg\n")
    # Match the loader's explicit TXT source selection unless JSON was requested.
    assert isinstance(create_validation_metric(validation, dataset), MeanAveragePrecision)
    validation.evaluator = "coco"
    assert isinstance(create_validation_metric(validation, dataset), CocoJsonEvaluator)
    validation.evaluator = "auto"
    validation.annotation_path = "annotations/instances_val.json"
    assert isinstance(create_validation_metric(validation, dataset), CocoJsonEvaluator)


def test_json_mode_rejects_augmentation_and_category_mismatch(configs):
    validation, dataset = configs
    validation.data.data_augment = {"HorizontalFlip": 0.5}
    with pytest.raises(ValueError, match="PadAndResize"):
        create_validation_metric(validation, dataset)
    validation.data.data_augment = {}
    dataset.class_num = 2
    with pytest.raises(ValueError, match="category count"):
        create_validation_metric(validation, dataset)


def test_training_initialization_selects_nested_validation_config(configs):
    validation, dataset = configs
    cfg = SimpleNamespace(task=SimpleNamespace(task="train", validation=validation), dataset=dataset,
                          model=None, weight=False)
    with patch("yolo.tools.solver.create_model", return_value=torch.nn.Identity()), patch("yolo.tools.solver.create_dataloader", return_value=[]):
        model = ValidateModel(cfg)
    assert model.validation_cfg is validation
    assert isinstance(model.metric, CocoJsonEvaluator)


def test_validation_uses_official_gt_and_resets_at_epoch_end(configs):
    validation, dataset = configs
    metric = create_validation_metric(validation, dataset)
    prediction = torch.tensor([[0., 10., 10., 30., 30., .9]])
    module = SimpleNamespace(metric=metric, ema=Mock(), post_process=Mock(return_value=[prediction]),
                             device=torch.device("cpu"), log_dict=Mock())
    # Loader GT intentionally has no boxes: official annotation must still be used.
    batch = (1, torch.zeros(1, 3, 100, 100), torch.zeros(1, 0, 5), None, ["sample.jpg"])
    outputs, batch_map = ValidateModel.validation_step(module, batch, 0)
    assert batch_map is None and outputs[0] is prediction
    assert metric.image_ids == [42]
    ValidateModel.on_validation_epoch_end(module)
    logged = module.log_dict.call_args_list[0]
    assert float(logged.args[0]["map"]) == pytest.approx(1.)
    assert logged.kwargs["sync_dist"] is False
    assert metric.image_ids == []
    # Second epoch must not retain the first epoch's predictions.
    module.post_process.return_value = [torch.zeros(0, 6)]
    ValidateModel.validation_step(module, batch, 0)
    ValidateModel.on_validation_epoch_end(module)
    assert float(module.log_dict.call_args_list[2].args[0]["map"]) == 0.


def test_training_sanity_validation_and_checkpoint_monitor_use_coco_json(configs, tmp_path):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from lightning import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from PIL import Image

    from yolo.tools.solver import TrainModel
    from yolo.utils.model_utils import EMA

    image_dir = tmp_path / "images" / "val"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), color=(80, 90, 100)).save(image_dir / "sample.jpg")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "yolo/config"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "task=train", "model=v9-t", "dataset=mock", "weight=false", "dataset.auto_download=null",
            f"dataset.path={tmp_path}", "dataset.train=val", "dataset.class_num=1",
            "image_size=[64,64]", "cpu_num=0", "task.data.batch_size=1", "task.validation.data.batch_size=1",
            "task.epoch=2", "use_wandb=false",
        ])
    cfg.task.data.data_augment = {}
    checkpoint = ModelCheckpoint(dirpath=tmp_path / "checkpoints", monitor="map", mode="max", save_last=True)
    model = TrainModel(cfg)
    trainer = Trainer(accelerator="cpu", devices=1, precision="32-true", max_epochs=2,
                      callbacks=[EMA(cfg.task.ema.decay), checkpoint], logger=False,
                      enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp_path)
    trainer.fit(model)
    assert isinstance(model.metric, CocoJsonEvaluator)
    assert trainer.current_epoch == 2
    assert torch.isfinite(trainer.callback_metrics["map"])
    assert Path(checkpoint.best_model_path).is_file()
    assert model.metric.image_ids == []
