import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import yolo.tools.yolov9_dataset as dataset_module
from yolo.tools.yolov9_dataset import YOLOv9Dataset


class IdentityYOLOv9Augmentation:
    def __init__(self, image_size, **kwargs):
        self.image_size = image_size
        self.kwargs = kwargs
        self.last_rgb_image = None
        self.sample = None

    def __call__(self, image, boxes, segments, get_sample):
        self.last_rgb_image = image.copy()
        self.sample = get_sample()
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255
        return tensor, torch.from_numpy(boxes.copy()), torch.arange(5, dtype=torch.float32)


@pytest.fixture(autouse=True)
def identity_augmentation(monkeypatch):
    monkeypatch.setattr(dataset_module, "YOLOv9Augmentation", IdentityYOLOv9Augmentation)


def _configs(root, phase="train"):
    data_cfg = SimpleNamespace(image_size=[8, 4], data_augment={"YOLOv9": {"mosaic": 1.0}})
    dataset_cfg = {"path": str(root), phase: phase, "class_num": 3}
    return SimpleNamespace(**data_cfg.__dict__), dataset_cfg


def _write_image(path, bgr=(7, 11, 23)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:] = bgr
    assert cv2.imwrite(str(path), image)


def test_txt_split_preserves_real_segments_and_box_only_entries(tmp_path):
    image_path = tmp_path / "images" / "train" / "mixed.png"
    empty_path = tmp_path / "images" / "train" / "empty.png"
    _write_image(image_path)
    _write_image(empty_path)
    (tmp_path / "train.txt").write_text("images/train/mixed.png\nimages/train/empty.png\n", encoding="utf-8")
    label_path = tmp_path / "labels" / "train" / "mixed.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text(
        "1 0.5 0.5 0.5 0.5\n" "0 0.125 0.25 0.375 0.25 0.375 0.75 0.125 0.75\n",
        encoding="utf-8",
    )

    data_cfg, dataset_cfg = _configs(tmp_path)
    dataset = YOLOv9Dataset(data_cfg, dataset_cfg, "train")

    assert len(dataset) == 2
    assert dataset.img_paths == [empty_path, image_path]
    assert dataset.bboxes[0].shape == (0, 5)
    assert dataset.segments[0] == []
    np.testing.assert_allclose(dataset.bboxes[1][0], [1, 0.25, 0.25, 0.75, 0.75])
    assert dataset.segments[1][0].shape == (0, 2)
    np.testing.assert_allclose(
        dataset.segments[1][1],
        [[0.125, 0.25], [0.375, 0.25], [0.375, 0.75], [0.125, 0.75]],
    )

    image, boxes, reverse, path = dataset[1]
    assert image.shape == (3, 4, 8)
    assert image.dtype == torch.float32
    assert image[:, 0, 0].tolist() == pytest.approx([23 / 255, 11 / 255, 7 / 255])
    assert boxes[0].tolist() == pytest.approx([1, 2, 1, 6, 3])
    assert reverse.tolist() == [0, 1, 2, 3, 4]
    assert path == image_path
    assert dataset.transform.last_rgb_image[0, 0].tolist() == [23, 11, 7]
    sample_image, sample_boxes, sample_segments = dataset.transform.sample
    assert sample_image.shape == (4, 8, 3)
    assert sample_boxes.shape[1] == 5
    assert len(sample_segments) == len(sample_boxes)


def test_coco_uses_authoritative_bbox_category_mapping_and_merged_polygons(tmp_path):
    first_path = tmp_path / "images" / "train" / "first.png"
    second_path = tmp_path / "images" / "train" / "second.png"
    _write_image(first_path)
    _write_image(second_path)
    annotation_path = tmp_path / "annotations" / "instances_train.json"
    annotation_path.parent.mkdir()
    annotation_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 9}, {"id": 2}],
                "images": [
                    {"id": 11, "file_name": "second.png", "width": 8, "height": 4},
                    {"id": 10, "file_name": "first.png", "width": 8, "height": 4},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "category_id": 9,
                        "iscrowd": 0,
                        "bbox": [1, 1, 4, 2],
                        "segmentation": [
                            [0, 0, 1, 0, 1, 1],
                            [4, 1, 7, 1, 7, 3, 4, 3],
                        ],
                    },
                    {
                        "id": 2,
                        "image_id": 10,
                        "category_id": 2,
                        "iscrowd": 0,
                        "bbox": [0, 0, 2, 2],
                    },
                    {
                        "id": 3,
                        "image_id": 10,
                        "category_id": 2,
                        "iscrowd": 1,
                        "bbox": [0, 0, 8, 4],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    # A legacy cache must have no effect on this loader.
    (tmp_path / "train.pache").write_bytes(b"not a torch cache")

    data_cfg, dataset_cfg = _configs(tmp_path)
    dataset = YOLOv9Dataset(data_cfg, dataset_cfg, "train")

    assert len(dataset) == 2  # includes the image with no annotations
    assert dataset.img_paths == [first_path, second_path]
    np.testing.assert_allclose(dataset.bboxes[0][0], [1, 0.125, 0.25, 0.625, 0.75])
    np.testing.assert_allclose(dataset.bboxes[0][1], [0, 0, 0, 0.25, 0.5])
    merged_points = dataset.segments[0][0]
    assert merged_points.shape[0] == 9
    assert any(np.allclose(point, [0, 0]) for point in merged_points)
    assert any(np.allclose(point, [0.875, 0.75]) for point in merged_points)
    # The nearest-point connector is traversed in both directions, so it does
    # not create the filled diagonal produced by naive list flattening.
    np.testing.assert_allclose(merged_points[0], merged_points[3])
    np.testing.assert_allclose(merged_points[4], merged_points[-1])
    assert dataset.segments[0][1].shape == (0, 2)  # bbox-only remains bbox-only
    assert dataset.bboxes[1].shape == (0, 5)


def test_txt_rejects_out_of_range_coordinates(tmp_path):
    image_path = tmp_path / "images" / "train" / "bad.png"
    _write_image(image_path)
    (tmp_path / "train.txt").write_text("images/train/bad.png\n", encoding="utf-8")
    label_path = tmp_path / "labels" / "train" / "bad.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 1.1 0.5 0.2 0.2\n", encoding="utf-8")

    data_cfg, dataset_cfg = _configs(tmp_path)
    with pytest.raises(ValueError, match=r"normalized|outside"):
        YOLOv9Dataset(data_cfg, dataset_cfg, "train")


def test_txt_directory_fallback_keeps_images_without_labels(tmp_path):
    labeled_path = tmp_path / "images" / "train" / "b.png"
    empty_path = tmp_path / "images" / "train" / "a.png"
    _write_image(labeled_path)
    _write_image(empty_path)
    label_path = tmp_path / "labels" / "train" / "b.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("2 0.5 0.5 1.0 1.0\n", encoding="utf-8")

    data_cfg, dataset_cfg = _configs(tmp_path)
    dataset = YOLOv9Dataset(data_cfg, dataset_cfg, "train")

    assert dataset.img_paths == [empty_path, labeled_path]
    assert dataset.bboxes[0].shape == (0, 5)
    np.testing.assert_allclose(dataset.bboxes[1], [[2, 0, 0, 1, 1]])
    assert dataset.segments[1][0].shape == (0, 2)
