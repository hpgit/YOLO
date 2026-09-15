"""Dataset adapter for the original YOLOv9 augmentation recipe.

This loader deliberately does not consume the legacy ``.pache`` files.  Those
files only retain bounding boxes, so using them would silently turn every
segmentation into box-only data and disable meaningful copy-paste.  Labels are
parsed from authoritative JSON or per-image TXT files when the dataset object
is created.  This adds a small startup cost in exchange for keeping boxes and
polygon contours aligned.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from yolo.tools.data_conversion import discretize_categories
from yolo.tools.yolov9_augmentation import YOLOv9Augmentation
from yolo.utils.annotation_utils import parse_coco_bbox, parse_yolo_label, polygon_area as _polygon_area


BoxArray = np.ndarray
SegmentList = List[np.ndarray]
RawSample = Tuple[np.ndarray, BoxArray, SegmentList]


class YOLOv9Dataset(Dataset):
    """Load normalized boxes and real polygons for YOLOv9 augmentation.

    A segment is aligned one-to-one with its box.  Box-only annotations use an
    empty ``(0, 2)`` segment instead of a synthetic rectangle.  COCO multipart
    annotations are joined through nearest-point, zero-width round-trip links.
    This keeps every component in the one-contour-per-object augmentation API
    without the filled bridge produced by simply concatenating point lists.
    """

    _IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

    def __init__(self, data_cfg: Any, dataset_cfg: Any, phase: str = "train2017") -> None:
        self.image_size = list(data_cfg.image_size)
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must contain width and height, got {self.image_size!r}")
        if getattr(data_cfg, "dynamic_shape", False):
            raise ValueError("YOLOv9 augmentation requires a fixed image_size; dynamic_shape is not supported")

        self.dataset_path = Path(_config_get(dataset_cfg, "path", ""))
        self.phase_name = _config_get(dataset_cfg, phase, phase)
        self.class_num = _config_get(dataset_cfg, "class_num", None)

        augment_cfg = _config_get(data_cfg.data_augment, "YOLOv9", None)
        if augment_cfg is None:
            raise ValueError("data_augment.YOLOv9 is required for YOLOv9Dataset")
        self.transform = YOLOv9Augmentation(self.image_size, **dict(augment_cfg))

        split_path = self.dataset_path / f"{self.phase_name}.txt"
        if split_path.is_file():
            self.img_paths, self.bboxes, self.segments = self._load_txt_split(split_path)
        else:
            annotation_path = self.dataset_path / "annotations" / f"instances_{self.phase_name}.json"
            images_path = self.dataset_path / "images" / str(self.phase_name)
            if annotation_path.is_file():
                self.img_paths, self.bboxes, self.segments = self._load_coco_json(annotation_path)
            elif images_path.is_dir():
                self.img_paths, self.bboxes, self.segments = self._load_txt_directory(images_path)
            else:
                raise FileNotFoundError(
                    f"Expected an explicit split file at '{split_path}' or COCO annotations at "
                    f"'{annotation_path}', or images with per-image TXT labels under '{images_path}'"
                )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Path]:
        image, boxes, segments = self._get_raw_sample(idx)
        image, boxes, rev_tensor = self.transform(image, boxes, segments, self.get_sample)

        image_tensor = _as_image_tensor(image)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5).clone()
        if boxes_tensor.numel():
            height, width = image_tensor.shape[-2:]
            boxes_tensor[:, [1, 3]] *= width
            boxes_tensor[:, [2, 4]] *= height

        rev_tensor = torch.as_tensor(rev_tensor, dtype=torch.float32).reshape(-1)
        if rev_tensor.numel() != 5:
            raise ValueError(f"YOLOv9 augmentation must return a five-element reverse tensor, got {rev_tensor.shape}")
        return image_tensor, boxes_tensor, rev_tensor, self.img_paths[idx]

    def get_sample(self, count: int = 1) -> RawSample | List[RawSample]:
        """Return raw random samples while matching YOLOv9's RNG call shape."""

        if not self.img_paths:
            raise IndexError("Cannot sample from an empty dataset")
        if count < 1:
            raise ValueError(f"Sample count must be positive, got {count}")
        if count == 1:
            return self._get_raw_sample(random.randint(0, len(self) - 1))
        indices = random.choices(range(len(self)), k=count)
        return [self._get_raw_sample(index) for index in indices]

    def _get_raw_sample(self, idx: int) -> RawSample:
        path = self.img_paths[idx]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = self.bboxes[idx].copy()
        segments = [segment.copy() for segment in self.segments[idx]]
        return image, boxes, segments

    def _load_txt_split(self, split_path: Path) -> Tuple[List[Path], List[BoxArray], List[SegmentList]]:
        with split_path.open(encoding="utf-8") as split_file:
            entries = [line.strip() for line in split_file if line.strip()]

        image_paths: List[Path] = []
        for entry in entries:
            image_path = Path(entry)
            if not image_path.is_absolute():
                image_path = self.dataset_path / image_path
            if image_path.suffix.lower() not in self._IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image path in {split_path}: {entry}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed by {split_path} does not exist: {image_path}")
            image_paths.append(image_path)

        return self._load_txt_images(image_paths)

    def _load_txt_directory(self, images_path: Path) -> Tuple[List[Path], List[BoxArray], List[SegmentList]]:
        image_paths = [
            path
            for path in images_path.iterdir()
            if path.is_file() and path.suffix.lower() in self._IMAGE_SUFFIXES
        ]
        return self._load_txt_images(image_paths)

    def _load_txt_images(self, image_paths: Sequence[Path]) -> Tuple[List[Path], List[BoxArray], List[SegmentList]]:
        img_paths: List[Path] = []
        boxes_per_image: List[BoxArray] = []
        segments_per_image: List[SegmentList] = []

        for image_path in image_paths:
            label_path = self._label_path_for_image(image_path)
            boxes, segments = self._read_yolo_label(label_path)
            img_paths.append(image_path)
            boxes_per_image.append(boxes)
            segments_per_image.append(segments)

        return _sort_samples(img_paths, boxes_per_image, segments_per_image)

    def _label_path_for_image(self, image_path: Path) -> Path:
        try:
            relative = image_path.relative_to(self.dataset_path)
        except ValueError:
            relative = image_path

        parts = list(relative.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            label_path = Path(*parts).with_suffix(".txt")
            return self.dataset_path / label_path if not label_path.is_absolute() else label_path
        return self.dataset_path / "labels" / str(self.phase_name) / f"{image_path.stem}.txt"

    def _read_yolo_label(self, label_path: Path) -> Tuple[BoxArray, SegmentList]:
        if not label_path.is_file():
            return _empty_boxes(), []

        boxes: List[List[float]] = []
        segments: SegmentList = []
        with label_path.open(encoding="utf-8") as label_file:
            for line_number, raw_line in enumerate(label_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                box, segment = parse_yolo_label(
                    line.split(), f"{label_path}:{line_number}", self.class_num
                )
                boxes.append(box)
                segments.append(segment)

        return _box_array(boxes), segments

    def _load_coco_json(self, annotation_path: Path) -> Tuple[List[Path], List[BoxArray], List[SegmentList]]:
        with annotation_path.open(encoding="utf-8") as annotation_file:
            data = json.load(annotation_file)

        categories = data.get("categories", [])
        category_map = discretize_categories(categories) if categories else None
        annotations: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for annotation in data.get("annotations", []):
            if not annotation.get("iscrowd", False):
                annotations[annotation.get("image_id")].append(annotation)

        img_paths: List[Path] = []
        boxes_per_image: List[BoxArray] = []
        segments_per_image: List[SegmentList] = []
        for image_info in data.get("images", []):
            image_path = self._resolve_json_image_path(str(image_info["file_name"]))
            if not image_path.is_file():
                raise FileNotFoundError(f"COCO image does not exist: {image_path}")
            width, height = float(image_info["width"]), float(image_info["height"])
            if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions for COCO image {image_info.get('id')}")

            boxes: List[List[float]] = []
            segments: SegmentList = []
            for annotation in annotations.get(image_info.get("id"), []):
                parsed = self._parse_coco_annotation(annotation, width, height, category_map)
                if parsed is None:
                    continue
                box, segment = parsed
                boxes.append(box)
                segments.append(segment)

            img_paths.append(image_path)
            boxes_per_image.append(_box_array(boxes))
            segments_per_image.append(segments)

        return _sort_samples(img_paths, boxes_per_image, segments_per_image)

    def _resolve_json_image_path(self, file_name: str) -> Path:
        source = Path(file_name)
        if source.is_absolute():
            return source
        candidates = (
            self.dataset_path / "images" / str(self.phase_name) / source,
            self.dataset_path / "images" / source,
            self.dataset_path / source,
        )
        return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])

    def _parse_coco_annotation(
        self,
        annotation: Dict[str, Any],
        image_width: float,
        image_height: float,
        category_map: Dict[int, int] | None,
    ) -> Tuple[List[float], np.ndarray] | None:
        box = parse_coco_bbox(annotation, image_width, image_height, category_map, self.class_num)
        if box is None:
            return None
        segment = _merge_coco_polygons(annotation.get("segmentation"), image_width, image_height)
        return box, segment


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _empty_boxes() -> BoxArray:
    return np.zeros((0, 5), dtype=np.float32)


def _sort_samples(
    paths: List[Path], boxes: List[BoxArray], segments: List[SegmentList]
) -> Tuple[List[Path], List[BoxArray], List[SegmentList]]:
    records = sorted(zip(paths, boxes, segments), key=lambda record: str(record[0]))
    if not records:
        return [], [], []
    sorted_paths, sorted_boxes, sorted_segments = zip(*records)
    return list(sorted_paths), list(sorted_boxes), list(sorted_segments)


def _box_array(boxes: Sequence[Sequence[float]]) -> BoxArray:
    if not boxes:
        return _empty_boxes()
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 5)


def _empty_segment() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float32)


def _merge_coco_polygons(segmentation: Any, width: float, height: float) -> np.ndarray:
    if not isinstance(segmentation, list) or not segmentation:
        return _empty_segment()
    polygons = segmentation if isinstance(segmentation[0], list) else [segmentation]
    valid_polygons: List[np.ndarray] = []
    for polygon in polygons:
        try:
            coordinates = np.asarray(polygon, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if coordinates.size < 6 or coordinates.size % 2 or not np.isfinite(coordinates).all():
            continue
        points = coordinates.reshape(-1, 2) / np.asarray([width, height], dtype=np.float32)
        points = np.clip(points, 0.0, 1.0)
        if abs(_polygon_area(points)) > 0:
            valid_polygons.append(points.astype(np.float32, copy=False))
    if not valid_polygons:
        return _empty_segment()
    if len(valid_polygons) == 1:
        return valid_polygons[0].copy()

    # Link consecutive contours at their nearest vertices.  Middle contours
    # are traversed in two pieces so every connecting edge is retraced in the
    # reverse direction.  The connector therefore has zero enclosed width.
    exits: List[int | None] = [None] * len(valid_polygons)
    entries: List[int | None] = [None] * len(valid_polygons)
    for index in range(len(valid_polygons) - 1):
        left, right = valid_polygons[index], valid_polygons[index + 1]
        distances = ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2)
        left_index, right_index = np.unravel_index(int(np.argmin(distances)), distances.shape)
        exits[index] = int(left_index)
        entries[index + 1] = int(right_index)

    merged: List[np.ndarray] = []
    middle_remainders: List[np.ndarray] = []
    for index, polygon in enumerate(valid_polygons):
        if index == 0:
            start = int(exits[index])
            rotated = np.roll(polygon, -start, axis=0)
            merged.append(np.concatenate((rotated, rotated[:1]), axis=0))
        elif index == len(valid_polygons) - 1:
            start = int(entries[index])
            rotated = np.roll(polygon, -start, axis=0)
            merged.append(np.concatenate((rotated, rotated[:1]), axis=0))
        else:
            start = int(entries[index])
            rotated = np.roll(polygon, -start, axis=0)
            exit_offset = (int(exits[index]) - start) % len(polygon)
            merged.append(rotated[: exit_offset + 1])
            middle_remainders.append(np.concatenate((rotated[exit_offset:], rotated[:1]), axis=0))

    merged.extend(reversed(middle_remainders))
    return np.concatenate(merged, axis=0).astype(np.float32, copy=False)


def _as_image_tensor(image: Any) -> Tensor:
    tensor = torch.as_tensor(image)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"YOLOv9 augmentation must return a CHW RGB image, got {tensor.shape}")
    tensor = tensor.to(dtype=torch.float32).contiguous()
    if tensor.numel() and tensor.max() > 1:
        tensor = tensor / 255.0
    if not torch.isfinite(tensor).all() or (tensor < 0).any() or (tensor > 1).any():
        raise ValueError("YOLOv9 augmentation image must be finite and scaled to [0, 1]")
    return tensor
