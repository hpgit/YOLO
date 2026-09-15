"""Detection-only annotations must reach loss and both evaluation backends."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning import Trainer
from omegaconf import OmegaConf
from PIL import Image

from yolo.tools.data_loader import YoloDataset, create_dataloader
from yolo.tools.solver import TrainModel, ValidateModel, create_validation_metric
from yolo.tools.yolov9_dataset import YOLOv9Dataset
from yolo.utils.bounding_box_utils import to_metrics_format
from yolo.utils.coco_eval import CocoJsonEvaluator
from yolo.utils.model_utils import EMA


def _write_dataset(root, source="txt", annotation_extra=None):
    images = root / "images" / "sample"
    images.mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (64, 32), (40 + index * 50, 80, 120)).save(images / f"{index}.png")
    if source in {"txt", "split"}:
        labels = root / "labels" / "sample"
        labels.mkdir(parents=True)
        # The asymmetric box detects accidental xywh-as-polygon parsing.
        for index in range(2):
            (labels / f"{index}.txt").write_text("\n0 0.375 0.5 0.5 0.5\n\n")
        if source == "split":
            (root / "sample.txt").write_text("\nimages/sample/0.png\nimages/sample/1.png\n")
    else:
        annotations = root / "annotations"
        annotations.mkdir()
        (annotations / "instances_sample.json").write_text(json.dumps({
            "images": [{"id": i, "file_name": f"{i}.png", "width": 64, "height": 32} for i in range(2)],
            "categories": [{"id": 7, "name": "object"}],
            "annotations": [dict(id=i + 1, image_id=i, category_id=7, bbox=[8, 8, 32, 16],
                                 **(annotation_extra or {})) for i in range(2)],
        }))


def _configs(root, reference=False):
    data = OmegaConf.create(dict(image_size=[64, 64], batch_size=2, cpu_num=0, shuffle=False,
                                 pin_memory=False, data_augment={}))
    if reference:
        data.data_augment = {"YOLOv9": dict(mosaic=0., mixup=0., translate=0., scale=0.,
                                           fliplr=0., hsv_h=0., hsv_s=0., hsv_v=0., albumentations=False)}
    dataset = OmegaConf.create(dict(path=str(root), train="sample", validation="sample", class_num=1))
    return data, dataset


@pytest.mark.parametrize("source", ["txt", "split", "json"])
@pytest.mark.parametrize("reference", [False, True])
def test_bbox_only_coordinates_survive_train_and_validation(tmp_path, source, reference):
    _write_dataset(tmp_path, source)
    data, dataset = _configs(tmp_path, reference)
    train = create_dataloader(data, dataset, "train")
    if reference:
        assert all(segment.shape == (0, 2) for segments in train.dataset.segments for segment in segments)
    expected = torch.tensor([[[0., 8., 24., 40., 40.]]] * 2)
    torch.testing.assert_close(next(iter(train))[2], expected)
    data.data_augment = {}
    batch = next(iter(create_dataloader(data, dataset, "validation")))
    torch.testing.assert_close(batch[2], expected)
    # A perfect synthetic prediction must yield AP=AR=1 in the selected backend.
    metric_cfg = OmegaConf.create(dict(task="validation", evaluator="auto", data={"data_augment": {}}))
    metric = create_validation_metric(metric_cfg, dataset)
    predictions = [torch.cat((target, torch.ones(1, 1)), dim=1) for target in batch[2]]
    if isinstance(metric, CocoJsonEvaluator):
        metric.update(predictions, batch[4], [64, 64])
    else:
        metric.update([to_metrics_format(p) for p in predictions], [to_metrics_format(t) for t in batch[2]])
    result = metric.compute()
    assert float(result["map"]) == pytest.approx(1.)
    assert float(result["mar_100"]) == pytest.approx(1.)


@pytest.mark.parametrize("extra", [
    {}, {"segmentation": []}, {"segmentation": None},
    {"segmentation": {"size": [32, 64], "counts": "unused"}},
    {"segmentation": [[0, 0, 4, 0, 4, 4]]},
])
def test_coco_bbox_takes_precedence_over_optional_segmentation(tmp_path, extra):
    _write_dataset(tmp_path, "json", extra)
    for reference in (False, True):
        data, dataset = _configs(tmp_path, reference)
        loader = create_dataloader(data, dataset, "train")
        np.testing.assert_allclose(loader.dataset.bboxes[0], [[0, .125, .25, .625, .75]])


@pytest.mark.parametrize("row, message", [
    ("0 0.5 0.5 0 0.2", "positive"),
    ("0 0.5 0.5 -0.1 0.2", "normalized"),
    ("0 nan 0.5 0.2 0.2", "finite"),
    ("0 0.5 0.5 inf 0.2", "finite"),
    ("0 1.1 0.5 0.2 0.2", "normalized"),
    ("-1 0.5 0.5 0.2 0.2", "non-negative integer"),
    ("0.5 0.5 0.5 0.2 0.2", "non-negative integer"),
    ("nan 0.5 0.5 0.2 0.2", "non-negative integer"),
    ("1 0.5 0.5 0.2 0.2", "class_num"),
    ("0 0.5 0.5 0.2", "Expected 5"),
    ("0 0 0 1 1 0.5 0.5", "zero area"),
    ("0 a 0.5 0.2 0.2", "Non-numeric"),
])
@pytest.mark.parametrize("reference", [False, True])
def test_invalid_txt_fails_with_source_line(tmp_path, row, message, reference):
    _write_dataset(tmp_path)
    label = tmp_path / "labels/sample/0.txt"
    label.write_text("\n" + row + "\n")
    data, dataset = _configs(tmp_path, reference)
    with pytest.raises(ValueError, match=message) as error:
        create_dataloader(data, dataset, "train")
    assert f"{label}:2" in str(error.value)


@pytest.mark.parametrize("reference", [False, True])
def test_mixed_labels_empty_images_and_partial_boxes(tmp_path, reference):
    _write_dataset(tmp_path)
    (tmp_path / "labels/sample/0.txt").write_text(
        "0 0.0 0.5 0.5 0.5\n0 0.125 0.25 0.625 0.25 0.625 0.75 0.125 0.75\n"
    )
    (tmp_path / "labels/sample/1.txt").unlink()
    data, dataset = _configs(tmp_path, reference)
    batch = next(iter(create_dataloader(data, dataset, "train")))
    torch.testing.assert_close(batch[2][0], torch.tensor([[0., 0., 24., 16., 40.], [0., 8., 24., 40., 40.]]))
    assert (batch[2][1, :, 0] == -1).all()


@pytest.mark.parametrize("bbox", [[], [0, 0, 0, 1], [0, 0, -1, 1], [0, 0, float("nan"), 1],
                                  [0, 0, float("inf"), 1], [70, 0, 5, 5], [0, 0, "bad", 1],
                                  [[0], [0], [1], [1]], None])
def test_invalid_coco_boxes_are_skipped_in_both_loaders(tmp_path, bbox):
    _write_dataset(tmp_path, "json")
    path = tmp_path / "annotations/instances_sample.json"
    annotations = json.loads(path.read_text())
    annotations["annotations"][0]["bbox"] = bbox
    path.write_text(json.dumps(annotations))
    for reference in (False, True):
        data, dataset = _configs(tmp_path, reference)
        loader = create_dataloader(data, dataset, "train")
        assert not (loader.dataset.bboxes[0][:, 0] >= 0).any()
        np.testing.assert_allclose(loader.dataset.bboxes[1], [[0, .125, .25, .625, .75]])


@pytest.mark.parametrize("bbox, expected", [
    ([-8, 8, 24, 40], [0, 0., .25, .25, 1.]),
    ([63, 31, .0001, .0001], [0, 63 / 64, 31 / 32, 63.0001 / 64, 31.0001 / 32]),
])
def test_coco_partial_and_tiny_positive_boxes(tmp_path, bbox, expected):
    _write_dataset(tmp_path, "json")
    path = tmp_path / "annotations/instances_sample.json"
    annotations = json.loads(path.read_text())
    annotations["annotations"][0]["bbox"] = bbox
    path.write_text(json.dumps(annotations))
    for reference in (False, True):
        data, dataset = _configs(tmp_path, reference)
        loader = create_dataloader(data, dataset, "train")
        np.testing.assert_allclose(loader.dataset.bboxes[0], [expected], atol=1e-7)


@pytest.mark.parametrize("reference", [False, True])
def test_coco_unknown_category_fails_with_annotation_id(tmp_path, reference):
    _write_dataset(tmp_path, "json")
    path = tmp_path / "annotations/instances_sample.json"
    annotations = json.loads(path.read_text())
    annotations["annotations"][0]["category_id"] = 99
    path.write_text(json.dumps(annotations))
    data, dataset = _configs(tmp_path, reference)
    with pytest.raises(ValueError, match="Unknown COCO category_id 99.*annotation 1"):
        create_dataloader(data, dataset, "train")


def test_old_bbox_cache_is_rebuilt_then_reused_and_class_config_revalidated(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    cache = tmp_path / "sample.pache"
    torch.save([(tmp_path / "images/sample/0.png", torch.zeros(1, 5), 1.)], cache)
    data, dataset = _configs(tmp_path)
    first = create_dataloader(data, dataset).dataset
    np.testing.assert_allclose(first.bboxes[0], [[0, .125, .25, .625, .75]])
    assert len(first) == 2
    with monkeypatch.context() as patch:
        patch.setattr(YoloDataset, "filter_data", lambda *args: pytest.fail("valid cache was not reused"))
        second = create_dataloader(data, dataset).dataset
    np.testing.assert_array_equal(first.bboxes, second.bboxes)
    dataset.class_num = 0
    with pytest.raises(ValueError, match="class_num"):
        create_dataloader(data, dataset)


def test_bbox_only_coco_defaults_preserve_supplied_metadata_and_source(tmp_path):
    _write_dataset(tmp_path, "json", {"area": 123., "iscrowd": 1})
    path = tmp_path / "annotations/instances_sample.json"
    before = path.read_bytes()
    metric = CocoJsonEvaluator(path)
    assert all(a["area"] == 123. and a["iscrowd"] == 1 for a in metric.coco_gt.anns.values())
    assert path.read_bytes() == before


@pytest.mark.parametrize("source", ["txt", "json"])
@pytest.mark.parametrize("reference", [False, True])
def test_bbox_only_real_training_and_standalone_validation(tmp_path, source, reference):
    _write_dataset(tmp_path, source)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "yolo/config"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "task=train", "model=v9-t", "dataset=mock", "weight=false", "dataset.auto_download=null",
            f"dataset.path={tmp_path}", "dataset.train=sample", "dataset.validation=sample", "dataset.class_num=1",
            "image_size=[64,64]", "cpu_num=0", "task.data.batch_size=2", "task.validation.data.batch_size=2",
            "task.epoch=1", "use_wandb=false",
        ])
    if not reference:
        cfg.task.data.data_augment = {}
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    model = TrainModel(cfg)
    assert isinstance(model.train_loader.dataset, YOLOv9Dataset if reference else YoloDataset)
    before = {key: value.detach().clone() for key, value in model.model.named_parameters()}
    trainer = Trainer(accelerator="cpu", devices=1, precision="32-true", max_epochs=1,
                      callbacks=[EMA(cfg.task.ema.decay)], logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp_path)
    trainer.fit(model)
    assert trainer.global_step == 1
    assert any(not torch.equal(before[key], value) for key, value in model.model.named_parameters())
    assert all(torch.isfinite(value).all() for value in model.model.parameters())
    assert all(torch.isfinite(value).all() for value in trainer.callback_metrics.values())
    assert trainer.callback_metrics["Loss/BoxLoss_epoch"] > 0
    assert trainer.callback_metrics["Loss/DFLLoss_epoch"] > 0
    assert "map" in trainer.callback_metrics and "mar_100" in trainer.callback_metrics
    # The independently instantiated validation task must also work.
    cfg.task = cfg.task.validation
    validation_model = ValidateModel(cfg)
    validation_model.model.load_state_dict(model.model.state_dict())
    results = trainer.validate(validation_model, verbose=False)
    assert np.isfinite(results[0]["map"]) and np.isfinite(results[0]["mar_100"])
