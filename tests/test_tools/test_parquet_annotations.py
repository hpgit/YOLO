"""Pandas Parquet pseudo-labels must preserve image grouping and bbox geometry."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision

from yolo.tools.data_loader import create_dataloader
from yolo.tools.solver import create_validation_metric
from yolo.utils.parquet_utils import (
    PARQUET_COLUMNS,
    load_parquet_annotations,
    resolve_parquet_annotation,
)


def _frame(root):
    for name in ("a", "b"):
        path = root / "pictures" / f"{name}.png"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (64, 32), (40, 80, 120)).save(path)
    # Nondefault, repeated pandas indices must not be interpreted as labels.
    return pd.DataFrame(
        [
            ["pictures/b.png", 0.001, 1, 0.375, 0.5, 0.5, 0.5],
            ["pictures/a.png", 0.99, 0, 0.375, 0.5, 0.5, 0.5],
            ["pictures/b.png", 0.9, 0, 0.75, 0.25, 0.25, 0.25],
        ],
        columns=PARQUET_COLUMNS,
        index=[4, 4, 2],
    )


def _write(root, frame, name="labels.parquet"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow")
    return path


def _configs(root, source="labels.parquet", reference=False):
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
        dict(path=str(root), train=source, validation=source, class_num=2, class_list=["first", "second"])
    )
    return data, dataset


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("source", ["labels.parquet", "absolute", "sample"])
def test_grouping_coordinates_paths_and_confidence(tmp_path, reference, source):
    frame = _frame(tmp_path)
    frame.loc[2, "image"] = str(tmp_path / "pictures/b.png")
    filename = "annotations/instances_sample.parquet" if source == "sample" else "labels.parquet"
    path = _write(tmp_path, frame, filename)
    source = str(path) if source == "absolute" else source
    data, dataset = _configs(tmp_path, source, reference)
    loader = create_dataloader(data, dataset)
    assert len(loader.dataset) == 2
    batch = next(iter(loader))
    assert list(batch[4]) == [tmp_path / "pictures/a.png", tmp_path / "pictures/b.png"]
    torch.testing.assert_close(batch[2][0, 0], torch.tensor([0.0, 8.0, 24.0, 40.0, 40.0]))
    assert batch[2][0, 1, 0] == -1  # Padding only, not a second annotation.
    torch.testing.assert_close(batch[2][1], torch.tensor([[1.0, 8.0, 24.0, 40.0, 40.0], [0.0, 40.0, 20.0, 56.0, 28.0]]))
    if reference:
        assert all(segment.shape == (0, 2) for segments in loader.dataset.segments for segment in segments)
    # Confidence neither removes the low-confidence row nor changes targets.
    frame["conf"] = [float("nan"), 0.0, 0.1]
    _write(tmp_path, frame, filename)
    second = next(iter(create_dataloader(data, dataset)))
    torch.testing.assert_close(second[2], batch[2])
    data.data_augment = {}
    validation = next(iter(create_dataloader(data, dataset, "validation")))
    torch.testing.assert_close(validation[2], batch[2])


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("image", None, "image must"),
        ("image", "", "image must"),
        ("image", "missing.png", "does not exist"),
        ("id_class", -1, "non-negative integer"),
        ("id_class", 0.5, "non-negative integer"),
        ("id_class", 0.999999999, "non-negative integer"),
        ("id_class", float("nan"), "non-negative integer"),
        ("id_class", 2, "class_num"),
        ("box_cx", float("nan"), "finite"),
        ("box_cy", float("inf"), "finite"),
        ("box_cx", 1.1, "normalized"),
        ("box_w", -0.1, "normalized"),
        ("box_w", 0.0, "positive"),
        ("box_h", 0.0, "positive"),
    ],
)
@pytest.mark.parametrize("reference", [False, True])
def test_invalid_rows_report_file_and_row(tmp_path, reference, column, value, message):
    frame = _frame(tmp_path).reset_index(drop=True)
    if column == "id_class":
        frame[column] = frame[column].astype(float)
    frame.loc[1, column] = value
    path = _write(tmp_path, frame)
    data, dataset = _configs(tmp_path, reference=reference)
    with pytest.raises((ValueError, FileNotFoundError), match=message) as error:
        create_dataloader(data, dataset)
    assert f"{path}:row 2" in str(error.value)


@pytest.mark.parametrize("column", PARQUET_COLUMNS)
def test_missing_columns_fail_clearly(tmp_path, column):
    path = _write(tmp_path, _frame(tmp_path).drop(columns=column))
    with pytest.raises(ValueError, match="Missing Parquet annotation columns") as error:
        load_parquet_annotations(path, tmp_path, 2)
    assert column in str(error.value)


def test_empty_and_missing_parquet_fail_clearly(tmp_path):
    path = _write(tmp_path, _frame(tmp_path).iloc[:0])
    with pytest.raises(ValueError, match="No annotation rows"):
        load_parquet_annotations(path, tmp_path, 2)
    with pytest.raises(FileNotFoundError, match="Parquet annotation file"):
        load_parquet_annotations(tmp_path / "missing.parquet", tmp_path, 2)


@pytest.mark.parametrize("reference", [False, True])
def test_mixed_multiple_inputs_and_fresh_annotations(tmp_path, reference):
    frame = _frame(tmp_path)
    _write(tmp_path, frame[frame.image == "pictures/a.png"], "first.parquet")
    _write(tmp_path, frame[frame.image == "pictures/b.png"], "second.parquet")
    (tmp_path / "legacy.txt").write_text("pictures/a.png\n")
    data, dataset = _configs(tmp_path, ["second.parquet", "first.parquet", "legacy"], reference)
    loader = create_dataloader(data, dataset)
    assert list(loader.dataset.img_paths) == [
        tmp_path / "pictures/b.png",
        tmp_path / "pictures/a.png",
        tmp_path / "pictures/a.png",
    ]
    # Replacement annotations and stale legacy caches must not hide new boxes.
    changed = frame[frame.image == "pictures/a.png"].copy()
    changed["box_cx"] = 0.5
    _write(tmp_path, changed, "first.parquet")
    torch.save({"metadata": {"version": 2}, "data": []}, tmp_path / "first.parquet.pache")
    reread = create_dataloader(data, dataset).dataset
    np.testing.assert_allclose(reread.bboxes[1][0], [0.0, 0.25, 0.25, 0.75, 0.75])


def test_discovery_preserves_existing_precedence(tmp_path):
    parquet = _write(tmp_path, _frame(tmp_path), "annotations/instances_sample.parquet")
    assert resolve_parquet_annotation(tmp_path, "sample") == parquet
    (tmp_path / "sample.txt").write_text("pictures/a.png\n")
    assert resolve_parquet_annotation(tmp_path, "sample") is None
    (tmp_path / "sample.txt").unlink()
    (tmp_path / "annotations/instances_sample.json").write_text("{}")
    assert resolve_parquet_annotation(tmp_path, "sample") is None
    assert resolve_parquet_annotation(tmp_path, "annotations/instances_sample.parquet") == parquet
    assert resolve_parquet_annotation(tmp_path, "missing.parquet") == tmp_path / "missing.parquet"


def test_validation_auto_and_explicit_backend_selection(tmp_path):
    _write(tmp_path, _frame(tmp_path), "annotations/instances_sample.parquet")
    _, dataset = _configs(tmp_path, "sample")
    cfg = OmegaConf.create(dict(task="validation", evaluator="auto", data={"data_augment": {}}))
    assert isinstance(create_validation_metric(cfg, dataset), MeanAveragePrecision)
    cfg.evaluator = "coco"
    with pytest.raises(ValueError, match="Parquet validation requires"):
        create_validation_metric(cfg, dataset)


def test_dynamic_shape_sorts_parquet_samples(tmp_path):
    _write(tmp_path, _frame(tmp_path))
    Image.new("RGB", (32, 64)).save(tmp_path / "pictures/a.png")
    data, dataset = _configs(tmp_path)
    data.dynamic_shape = True
    loader = create_dataloader(data, dataset)
    assert list(loader.dataset.ratios) == [2.0, 0.5]
    assert next(iter(loader))[1].shape[0] == 2


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("image_entry", ["a.png", "sample/a.png", "images/sample/a.png", "absolute"])
@pytest.mark.parametrize("source", ["sample", "annotations/instances_sample.parquet"])
def test_image_paths_follow_coco_json_convention(tmp_path, reference, image_entry, source):
    image = tmp_path / "images/sample/a.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 32)).save(image)
    entry = str(image) if image_entry == "absolute" else image_entry
    frame = pd.DataFrame([[entry, 0.01, 1, 0.375, 0.5, 0.5, 0.5]], columns=PARQUET_COLUMNS)
    _write(tmp_path, frame, "annotations/instances_sample.parquet")
    data, dataset = _configs(tmp_path, source, reference)
    loaded = create_dataloader(data, dataset).dataset
    assert list(loaded.img_paths) == [image]


def test_split_image_path_takes_precedence_over_root_fallbacks(tmp_path):
    frame = pd.DataFrame([["a.png", 0.9, 0, 0.5, 0.5, 0.5, 0.5]], columns=PARQUET_COLUMNS)
    for relative in ["images/sample/a.png", "images/a.png", "a.png"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 32)).save(path)
    path = _write(tmp_path, frame, "annotations/instances_sample.parquet")
    assert load_parquet_annotations(path, tmp_path, 2)[0][0] == tmp_path / "images/sample/a.png"


def test_column_order_extra_fields_and_nullable_values(tmp_path):
    frame = _frame(tmp_path)
    frame["id_class"] = frame["id_class"].astype("Int64")
    frame["unused"] = "not a label"
    frame = frame[list(reversed(frame.columns))]
    path = _write(tmp_path, frame)
    samples = load_parquet_annotations(path, tmp_path, 2)
    assert len(samples) == 2
    assert samples[1][1].shape == (2, 5)


@pytest.mark.parametrize("reference", [False, True])
def test_class_indices_are_used_directly_without_category_mapping(tmp_path, reference):
    _write(tmp_path, _frame(tmp_path))
    data, dataset = _configs(tmp_path, reference=reference)
    dataset.class_list = ["dog", "person"]
    loaded = create_dataloader(data, dataset).dataset
    assert loaded.bboxes[0][0, 0] == 0
    assert loaded.bboxes[1][0, 0] == 1
    assert loaded.bboxes[1][1, 0] == 0


@pytest.mark.parametrize("reference", [False, True])
def test_all_pseudo_boxes_reach_batch_without_legacy_100_box_cap(tmp_path, reference):
    frame = _frame(tmp_path).iloc[[1]]
    _write(tmp_path, pd.concat([frame] * 125, ignore_index=True))
    data, dataset = _configs(tmp_path, reference=reference)
    batch = next(iter(create_dataloader(data, dataset)))
    assert batch[2].shape == (1, 125, 5)
    assert (batch[2][0, :, 0] == 0).all()


def test_explicit_coco_gt_with_parquet_input_uses_correct_image_root(tmp_path):
    import json

    from yolo.utils.coco_eval import CocoJsonEvaluator

    frame = _frame(tmp_path)
    path = _write(tmp_path, frame, "annotations/instances_sample.parquet")
    annotation = tmp_path / "authoritative.json"
    annotation.write_text(
        json.dumps(
            dict(
                images=[dict(id=1, file_name="a.png", width=64, height=32)],
                categories=[dict(id=1, name="first"), dict(id=18, name="second")],
                annotations=[],
            )
        )
    )
    _, dataset = _configs(tmp_path, str(path))
    cfg = OmegaConf.create(
        dict(task="validation", evaluator="coco", annotation_path=str(annotation), data={"data_augment": {}})
    )
    metric = create_validation_metric(cfg, dataset)
    assert isinstance(metric, CocoJsonEvaluator)
    assert metric.image_root == tmp_path / "images/sample"
