"""COCO person-keypoint dataset with detection-compatible pose targets.

The loader reads the official ``person_keypoints_<split>.json`` files directly.
It deliberately creates no cache beside the dataset, which keeps mounted or
shared COCO trees read-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Mapping

import torch
from lightning.fabric.utilities.seed import pl_worker_init_function
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

# COCO's left/right pairs, using zero-based indices in the standard 17-point
# order. Nose (0) is intentionally unchanged.
COCO_KEYPOINT_FLIP_INDEX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


def _config_value(config, key, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _phase_fraction(dataset_cfg, task: str, split: str) -> float:
    configured = _config_value(dataset_cfg, "fraction", 1.0)
    if isinstance(configured, Mapping) or hasattr(configured, "get"):
        configured = configured.get(task, configured.get(split, 1.0))
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        raise ValueError("dataset.fraction must be a number in (0, 1]")
    fraction = float(configured)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("dataset.fraction must be finite and in (0, 1]")
    return fraction


def _pose_flip_probability(data_cfg, task: str) -> float:
    augment = _config_value(data_cfg, "data_augment", {}) or {}
    keys = set(augment.keys())
    unsupported = keys - {"Pose"}
    if unsupported:
        raise ValueError(f"Pose data only supports data_augment.Pose.fliplr; got {sorted(unsupported)}")
    pose = augment.get("Pose", {}) if hasattr(augment, "get") else {}
    pose = pose or {}
    unsupported_pose = set(pose.keys()) - {"fliplr"}
    if unsupported_pose:
        raise ValueError(f"Unsupported Pose augmentation options: {sorted(unsupported_pose)}")
    probability = pose.get("fliplr", 0.0)
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("data_augment.Pose.fliplr must be a number in [0, 1]")
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("data_augment.Pose.fliplr must be finite and in [0, 1]")
    if task != "train" and probability != 0:
        raise ValueError("Pose horizontal flip is only supported for training")
    return probability


class CocoPoseDataset(Dataset):
    """COCO person pose samples as ``[cls, xyxy, (x, y, v) * K]`` tensors.

    Coordinates returned by :meth:`__getitem__` are absolute pixels in the
    letterboxed image. People with no labeled keypoints remain as detection
    targets with zeroed keypoints. Crowd annotations are excluded from training.
    """

    def __init__(self, data_cfg, dataset_cfg, task: str):
        if task not in {"train", "validation"}:
            raise ValueError(f"Pose dataset task must be 'train' or 'validation', got {task!r}")
        if bool(_config_value(data_cfg, "dynamic_shape", False)):
            raise ValueError("Pose data does not support dynamic_shape; use a fixed letterbox size")

        self.task = task
        self.training = task == "train"
        self.image_size = tuple(int(value) for value in _config_value(data_cfg, "image_size"))
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("Pose image_size must contain positive [width, height]")
        self.fliplr = _pose_flip_probability(data_cfg, task)
        self.num_keypoints = int(_config_value(dataset_cfg, "num_keypoints", 17))
        if self.num_keypoints != 17:
            raise ValueError("COCO pose annotations require dataset.num_keypoints=17")
        if int(_config_value(dataset_cfg, "class_num", 1)) != 1:
            raise ValueError("COCO person pose data requires dataset.class_num=1")

        self.root = Path(_config_value(dataset_cfg, "path"))
        self.split = str(_config_value(dataset_cfg, task, "train2017" if self.training else "val2017"))
        self.annotation_path = self.root / "annotations" / f"person_keypoints_{self.split}.json"
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"COCO pose annotations not found: {self.annotation_path}")

        with self.annotation_path.open("r", encoding="utf-8") as annotation_file:
            payload = json.load(annotation_file)
        self._load_payload(payload, dataset_cfg)

    def _load_payload(self, payload: dict, dataset_cfg) -> None:
        categories = [category for category in payload.get("categories", []) if category.get("name") == "person"]
        if len(categories) != 1:
            raise ValueError("COCO pose annotations must contain exactly one 'person' category")
        category = categories[0]
        self.person_category_id = int(category["id"])
        self.keypoint_names = tuple(category.get("keypoints", ()))
        if len(self.keypoint_names) != self.num_keypoints:
            raise ValueError(f"Expected {self.num_keypoints} COCO keypoint names, found {len(self.keypoint_names)}")
        self.skeleton = tuple((int(a) - 1, int(b) - 1) for a, b in category.get("skeleton", ()))

        image_by_id = {}
        for image in payload.get("images", []):
            image_id = int(image["id"])
            if image_id in image_by_id:
                raise ValueError(f"Duplicate COCO image id: {image_id}")
            width, height = image.get("width"), image.get("height")
            if (
                isinstance(width, bool)
                or isinstance(height, bool)
                or not isinstance(width, (int, float))
                or not isinstance(height, (int, float))
                or not math.isfinite(float(width))
                or not math.isfinite(float(height))
                or width <= 0
                or height <= 0
            ):
                raise ValueError(f"Image {image_id} has invalid dimensions: {width}x{height}")
            file_name = Path(str(image.get("file_name", "")))
            if not file_name.name or file_name.is_absolute() or ".." in file_name.parts:
                raise ValueError(f"Image {image_id} has an unsafe file_name: {file_name}")
            image_by_id[image_id] = {
                "id": image_id,
                "file_name": file_name,
                "width": float(width),
                "height": float(height),
            }
        if not image_by_id:
            raise ValueError(f"No images found in {self.annotation_path}")

        all_image_ids = sorted(image_by_id)
        fraction = _phase_fraction(dataset_cfg, self.task, self.split)
        subset_seed = int(_config_value(dataset_cfg, "subset_seed", 10))
        selected_count = min(len(all_image_ids), max(1, math.ceil(len(all_image_ids) * fraction)))
        if selected_count == len(all_image_ids):
            self.image_ids = all_image_ids
        else:
            self.image_ids = sorted(random.Random(subset_seed).sample(all_image_ids, selected_count))
        selected = set(self.image_ids)

        targets_by_image = {image_id: [] for image_id in self.image_ids}
        for annotation in payload.get("annotations", []):
            image_id = int(annotation.get("image_id", -1))
            if image_id not in selected or int(annotation.get("category_id", -1)) != self.person_category_id:
                continue
            if self.training and bool(annotation.get("iscrowd", 0)):
                continue
            targets_by_image[image_id].append(self._annotation_row(annotation, image_by_id[image_id]))

        target_width = 5 + self.num_keypoints * 3
        self.records = []
        for image_id in self.image_ids:
            image = image_by_id[image_id]
            rows = targets_by_image[image_id]
            targets = torch.tensor(rows, dtype=torch.float32).reshape(-1, target_width)
            self.records.append(
                {
                    **image,
                    "path": self.root / "images" / self.split / image["file_name"],
                    "targets": targets,
                }
            )

        digest = hashlib.sha256(",".join(map(str, self.image_ids)).encode("ascii")).hexdigest()
        self.manifest = {
            "annotation_path": str(self.annotation_path),
            "split": self.split,
            "fraction": fraction,
            "subset_seed": subset_seed,
            "total_images": len(all_image_ids),
            "selected_images": len(self.image_ids),
            "image_ids_sha256": digest,
        }

    def _annotation_row(self, annotation: dict, image: dict) -> list[float]:
        annotation_id = annotation.get("id", "unknown")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Annotation {annotation_id} must contain bbox [x, y, width, height]")
        try:
            x, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Annotation {annotation_id} bbox must be numeric") from error
        if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
            raise ValueError(f"Annotation {annotation_id} has a non-finite or non-positive bbox")
        image_width, image_height = image["width"], image["height"]
        tolerance = 1e-3
        outside_image = (
            x < -tolerance
            or y < -tolerance
            or x + width > image_width + tolerance
            or y + height > image_height + tolerance
        )
        if outside_image:
            raise ValueError(f"Annotation {annotation_id} bbox is outside image {image['id']} bounds")
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(image_width, x + width), min(image_height, y + height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Annotation {annotation_id} has an empty bbox after bounds validation")

        raw_keypoints = annotation.get("keypoints", [0.0] * (self.num_keypoints * 3))
        if not isinstance(raw_keypoints, list) or len(raw_keypoints) != self.num_keypoints * 3:
            raise ValueError(f"Annotation {annotation_id} must contain {self.num_keypoints * 3} keypoint values")
        keypoints = []
        visible_count = 0
        for index in range(self.num_keypoints):
            try:
                point_x, point_y, visibility = map(float, raw_keypoints[index * 3 : index * 3 + 3])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Annotation {annotation_id} keypoint {index} must be numeric") from error
            if not all(math.isfinite(value) for value in (point_x, point_y, visibility)):
                raise ValueError(f"Annotation {annotation_id} keypoint {index} is non-finite")
            if visibility not in (0.0, 1.0, 2.0):
                raise ValueError(f"Annotation {annotation_id} keypoint {index} visibility must be 0, 1, or 2")
            if visibility == 0:
                point_x = point_y = 0.0
            else:
                if not 0 <= point_x <= image_width or not 0 <= point_y <= image_height:
                    raise ValueError(f"Annotation {annotation_id} keypoint {index} is outside image bounds")
                visible_count += 1
            keypoints.extend((point_x, point_y, visibility))
        declared_count = int(annotation.get("num_keypoints", visible_count))
        if declared_count != visible_count:
            raise ValueError(
                f"Annotation {annotation_id} num_keypoints={declared_count} does not match "
                f"{visible_count} labeled points"
            )
        return [0.0, x1, y1, x2, y2, *keypoints]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = record["path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO pose image not found: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if image.size != (int(record["width"]), int(record["height"])):
            raise ValueError(
                f"COCO image size mismatch for {image_path}: JSON says "
                f"{int(record['width'])}x{int(record['height'])}, file is {image.width}x{image.height}"
            )
        targets = record["targets"].clone()
        if self.training and self.fliplr and torch.rand(()) < self.fliplr:
            image, targets = self._horizontal_flip(image, targets)
        image, targets, reverse = self._letterbox(image, targets)
        return TF.to_tensor(image), targets, reverse, str(image_path)

    def _horizontal_flip(self, image: Image.Image, targets: torch.Tensor):
        width = image.width
        image = TF.hflip(image)
        if not targets.numel():
            return image, targets
        old_x1 = targets[:, 1].clone()
        old_x2 = targets[:, 3].clone()
        targets[:, 1] = width - old_x2
        targets[:, 3] = width - old_x1
        keypoints = targets[:, 5:].reshape(-1, self.num_keypoints, 3)
        visible = keypoints[..., 2] > 0
        keypoints[..., 0] = torch.where(visible, width - keypoints[..., 0], keypoints[..., 0])
        keypoints = keypoints[:, COCO_KEYPOINT_FLIP_INDEX, :]
        invisible = keypoints[..., 2] == 0
        keypoints[..., 0] = keypoints[..., 0].masked_fill(invisible, 0)
        keypoints[..., 1] = keypoints[..., 1].masked_fill(invisible, 0)
        targets[:, 5:] = keypoints.reshape(targets.size(0), -1)
        return image, targets

    def _letterbox(self, image: Image.Image, targets: torch.Tensor):
        target_width, target_height = self.image_size
        width, height = image.size
        scale = min(target_width / width, target_height / height)
        resized_width, resized_height = int(width * scale), int(height * scale)
        resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        pad_left = (target_width - resized_width) // 2
        pad_top = (target_height - resized_height) // 2
        output = Image.new("RGB", (target_width, target_height), (114, 114, 114))
        output.paste(resized, (pad_left, pad_top))

        if targets.numel():
            scale_x, scale_y = resized_width / width, resized_height / height
            targets[:, [1, 3]] = targets[:, [1, 3]] * scale_x + pad_left
            targets[:, [2, 4]] = targets[:, [2, 4]] * scale_y + pad_top
            keypoints = targets[:, 5:].reshape(-1, self.num_keypoints, 3)
            visible = keypoints[..., 2] > 0
            keypoints[..., 0] = torch.where(visible, keypoints[..., 0] * scale_x + pad_left, keypoints[..., 0])
            keypoints[..., 1] = torch.where(visible, keypoints[..., 1] * scale_y + pad_top, keypoints[..., 1])
        # Integer resize dimensions can make the x/y gains differ slightly.
        # Preserve both exact gains so prediction and evaluation inversion is exact.
        reverse = torch.tensor([resized_width / width, resized_height / height, pad_left, pad_top], dtype=torch.float32)
        return output, targets, reverse


def pose_collate_fn(batch):
    """Pad variable instance counts while preserving the standard batch tuple."""

    batch_size = len(batch)
    target_width = batch[0][1].size(1)
    target_limit = max(item[1].size(0) for item in batch)
    batch_targets = torch.zeros(batch_size, target_limit, target_width, dtype=torch.float32)
    batch_targets[:, :, 0] = -1
    for index, (_, targets, _, _) in enumerate(batch):
        batch_targets[index, : targets.size(0)] = targets
    images, _, reverse, paths = zip(*batch)
    return batch_size, torch.stack(images), batch_targets, torch.stack(reverse), paths


def create_pose_dataloader(data_cfg, dataset_cfg, task: str):
    """Build a reproducibly seeded pose DataLoader without writing dataset caches."""

    dataset = CocoPoseDataset(data_cfg, dataset_cfg, task)
    return DataLoader(
        dataset,
        batch_size=int(_config_value(data_cfg, "batch_size")),
        shuffle=bool(_config_value(data_cfg, "shuffle", task == "train")),
        num_workers=int(_config_value(data_cfg, "cpu_num", 0)),
        pin_memory=bool(_config_value(data_cfg, "pin_memory", False)),
        generator=torch.Generator().manual_seed(torch.initial_seed()),
        worker_init_fn=pl_worker_init_function,
        collate_fn=pose_collate_fn,
    )


__all__ = [
    "COCO_KEYPOINT_FLIP_INDEX",
    "CocoPoseDataset",
    "create_pose_dataloader",
    "pose_collate_fn",
]
