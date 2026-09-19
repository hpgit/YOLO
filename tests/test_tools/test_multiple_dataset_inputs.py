"""Multiple configured splits must reach training and aggregate validation."""

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from yolo.tools.data_loader import YoloDataset, create_dataloader
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.tools.solver import create_validation_metric
from yolo.utils.bounding_box_utils import to_metrics_format
from yolo.utils.coco_eval import CocoJsonEvaluator


def _configs(root, reference=False):
    data = OmegaConf.create(
        dict(image_size=[64, 64], batch_size=2, cpu_num=0, shuffle=False, pin_memory=False, data_augment={})
    )
    if reference:
        data.data_augment = {
            "YOLOv9": dict(
                mosaic=0.0,
                mixup=0.0,
                translate=0.0,
                scale=0.0,
                fliplr=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                albumentations=False,
            )
        }
    dataset = OmegaConf.create(
        dict(path=str(root), train=["first", "second"], validation=["first", "second"], class_num=1)
    )
    return data, dataset


def _write_split(root, split, kind="txt", size=(64, 32)):
    image = root / "images" / split / "same.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", size, (50, 80, 120)).save(image)
    if kind == "json":
        annotation = root / "annotations" / f"instances_{split}.json"
        annotation.parent.mkdir(exist_ok=True)
        annotation.write_text(
            json.dumps(
                dict(
                    images=[dict(id=1, file_name="same.png", width=size[0], height=size[1])],
                    categories=[dict(id=7, name="object")],
                    annotations=[
                        dict(id=1, image_id=1, category_id=7, bbox=[size[0] / 4, size[1] / 4, size[0] / 2, size[1] / 2])
                    ],
                )
            )
        )
    else:
        label = root / "labels" / split / "same.txt"
        label.parent.mkdir(parents=True)
        label.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
        if kind == "split":
            (root / f"{split}.txt").write_text(f"images/{split}/same.png\n")
    return image


@pytest.mark.parametrize("kinds", [("txt", "txt"), ("split", "json"), ("json", "json")])
@pytest.mark.parametrize("reference", [False, True])
def test_multiple_inputs_reach_train_and_validation(tmp_path, kinds, reference):
    paths = [_write_split(tmp_path, split, kind) for split, kind in zip(("first", "second"), kinds)]
    data, dataset = _configs(tmp_path, reference)
    train = create_dataloader(data, dataset, "train")
    batch = next(iter(train))
    assert list(batch[4]) == paths
    assert batch[1].shape == (2, 3, 64, 64)
    torch.testing.assert_close(batch[2], torch.tensor([[[0.0, 16.0, 24.0, 48.0, 40.0]]] * 2))
    if reference:
        assert all(len(segments[0]) == 4 for segments in train.dataset.segments if len(segments[0]))
        # Augmentation sampling sees the combined dataset, including split 2.
        assert len(train.dataset.get_sample(8)) == 8
    data.data_augment = {}
    batch = next(iter(create_dataloader(data, dataset, "validation")))
    metric_cfg = OmegaConf.create(dict(task="validation", evaluator="auto", data={"data_augment": {}}))
    metric = create_validation_metric(metric_cfg, dataset)
    predictions = [torch.cat((target, torch.ones(1, 1)), dim=1) for target in batch[2]]
    if isinstance(metric, CocoJsonEvaluator):
        assert len(metric.coco_gt.imgs) == 2
        assert len(metric.coco_gt.anns) == 2
        metric.update(predictions, batch[4], [64, 64])
    else:
        metric.update([to_metrics_format(p) for p in predictions], [to_metrics_format(t) for t in batch[2]])
    assert float(metric.compute()["map"]) == pytest.approx(1.0)


def test_dynamic_shape_sorts_combined_data_and_reuses_each_cache(tmp_path, monkeypatch):
    _write_split(tmp_path, "first", size=(32, 64))
    _write_split(tmp_path, "second", size=(64, 32))
    data, dataset = _configs(tmp_path)
    data.dynamic_shape = True
    first = create_dataloader(data, dataset).dataset
    assert list(first.ratios) == [2.0, 0.5]
    batch = next(iter(create_dataloader(data, dataset)))
    assert batch[1].shape[0] == 2
    monkeypatch.setattr(YoloDataset, "filter_data", lambda *args: pytest.fail("cache was not reused"))
    second = create_dataloader(data, dataset).dataset
    np.testing.assert_array_equal(first.bboxes, second.bboxes)
    assert (tmp_path / "first.pache").is_file()
    assert (tmp_path / "second.pache").is_file()


@pytest.mark.parametrize("value", [[], None, 12, ["first", 4], [""], [["first"]]])
@pytest.mark.parametrize("reference", [False, True])
def test_invalid_inputs_have_actionable_errors(tmp_path, value, reference):
    data, dataset = _configs(tmp_path, reference)
    dataset.train = value
    with pytest.raises(ValueError, match=r"dataset.train.*string"):
        create_dataloader(data, dataset)


def test_single_item_list_preserves_coco_backend(tmp_path):
    _write_split(tmp_path, "first", "json")
    data, dataset = _configs(tmp_path)
    dataset.validation = ["first"]
    metric_cfg = OmegaConf.create(dict(task="validation", evaluator="auto", data={"data_augment": {}}))
    assert isinstance(create_validation_metric(metric_cfg, dataset), CocoJsonEvaluator)


def test_multiple_coco_inputs_reject_inconsistent_class_mapping(tmp_path):
    for split in ("first", "second"):
        _write_split(tmp_path, split, "json")
    annotation = tmp_path / "annotations/instances_second.json"
    content = json.loads(annotation.read_text())
    content["categories"][0]["name"] = "different"
    annotation.write_text(json.dumps(content))
    _, dataset = _configs(tmp_path)
    metric_cfg = OmegaConf.create(dict(task="validation", evaluator="auto", data={"data_augment": {}}))
    with pytest.raises(ValueError, match="same category"):
        create_validation_metric(metric_cfg, dataset)


def test_auto_download_selects_all_inputs_once(tmp_path, monkeypatch):
    _, dataset = _configs(tmp_path)
    dataset.auto_download = dict(images=dict(base_url="https://example.invalid/", first={}, second={}, unused={}))
    downloaded = []
    monkeypatch.setattr("yolo.tools.dataset_preparation.download_file", lambda url, path: downloaded.append(path.name))
    monkeypatch.setattr("yolo.tools.dataset_preparation.unzip_file", lambda *args: None)
    monkeypatch.setattr("yolo.tools.dataset_preparation.check_files", lambda *args: False)
    prepare_dataset(dataset, "train")
    assert downloaded == ["first.zip", "second.zip"]


@pytest.mark.parametrize("reference", [False, True])
def test_multiple_inputs_real_training_and_standalone_validation(tmp_path, reference):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from lightning import Trainer

    from yolo.tools.solver import TrainModel, ValidateModel

    for split in ("first", "second"):
        _write_split(tmp_path, split, "json")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "yolo/config"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=train",
                "model=v9-t",
                "dataset=mock",
                "weight=false",
                "dataset.auto_download=null",
                f"dataset.path={tmp_path}",
                "dataset.train=[first,second]",
                "dataset.validation=[first,second]",
                "dataset.class_num=1",
                "image_size=[64,64]",
                "cpu_num=0",
                "task.data.batch_size=2",
                "task.validation.data.batch_size=2",
                "task.epoch=1",
                "use_wandb=false",
            ],
        )
    if not reference:
        cfg.task.data.data_augment = {}
    model = TrainModel(cfg)
    before = {key: value.detach().clone() for key, value in model.model.named_parameters()}
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model)
    assert trainer.global_step == 1
    assert any(not torch.equal(before[key], value) for key, value in model.model.named_parameters())
    assert all(torch.isfinite(value).all() for value in trainer.callback_metrics.values())
    assert len(model.train_loader.dataset) == len(model.val_loader.dataset) == 2
    cfg.task = cfg.task.validation
    validation = ValidateModel(cfg)
    validation.model.load_state_dict(model.model.state_dict())
    results = trainer.validate(validation, verbose=False)
    assert np.isfinite(results[0]["map"])
    assert len(validation.metric.coco_gt.imgs) == 2
