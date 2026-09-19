"""Training augmentation used by the original YOLOv9 recipe.

This module operates at the dataset boundary: images are RGB uint8 arrays and
boxes are ``[class, x1, y1, x2, y2]`` in normalized coordinates.  The mosaic
and perspective stages use pixel coordinates internally and return a square
float tensor without a later pad/resize transform.
"""

import math
import random
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

Sample = Tuple[np.ndarray, np.ndarray, Sequence[np.ndarray]]
SampleGetter = Callable[[int], object]


_DEFAULT_HYP = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.9,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.3,
    "albumentations": True,
}


def _probability(value: object, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("{} must be between 0 and 1, got {}".format(name, value))
    return value


def _as_boxes(boxes: object) -> np.ndarray:
    if boxes is None:
        return np.zeros((0, 5), dtype=np.float32)
    if isinstance(boxes, torch.Tensor):
        if boxes.device.type != "cpu":
            raise ValueError("YOLOv9 augmentation boxes must be on CPU")
        boxes = boxes.detach().numpy()
    result = np.asarray(boxes, dtype=np.float32)
    if result.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 5:
        raise ValueError("boxes must have shape (N, 5) as class and normalized xyxy")
    if not np.isfinite(result).all():
        raise ValueError("boxes contain a non-finite value")
    return result.copy()


def _as_segments(segments: object, count: int) -> List[np.ndarray]:
    if segments is None or (isinstance(segments, (list, tuple)) and not segments):
        return [np.zeros((0, 2), dtype=np.float32) for _ in range(count)]
    if len(segments) != count:
        raise ValueError("segments must be empty or contain one polygon per box")

    result = []
    for polygon in segments:
        polygon = np.asarray(polygon, dtype=np.float32)
        if polygon.size == 0:
            result.append(np.zeros((0, 2), dtype=np.float32))
            continue
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError("each segment must have shape (N, 2)")
        if not np.isfinite(polygon).all():
            raise ValueError("segments contain a non-finite value")
        if (polygon < 0).any() or (polygon > 1).any():
            raise ValueError("segment coordinates must be normalized to [0, 1]")
        result.append(polygon.copy())
    return result


def _validate_image(image: object) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array with shape (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image dimensions must be non-zero")
    return np.ascontiguousarray(image)


def _resize_longest(image: np.ndarray, size: int) -> np.ndarray:
    """Resize so the longest side is ``size``, using YOLOv9 floor rounding."""
    height, width = image.shape[:2]
    ratio = size / max(height, width)
    if ratio == 1.0:
        return image
    resized_width = max(1, int(width * ratio))
    resized_height = max(1, int(height * ratio))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)


def _bbox_ioa(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    left = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    top = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    right = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    bottom = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    intersection = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return intersection / (area2[None, :] + 1e-7)


def _copy_paste(
    image: np.ndarray,
    labels: np.ndarray,
    segments: List[np.ndarray],
    probability: float,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Mirror a random subset of non-overlapping segmented objects."""
    if probability == 0.0 or not segments or len(labels) == 0:
        return image, labels, segments

    height, width = image.shape[:2]
    mirrored_boxes = labels[:, 1:5].copy()
    mirrored_boxes[:, [0, 2]] = width - labels[:, [3, 1]]
    eligible = (_bbox_ioa(mirrored_boxes, labels[:, 1:5]) < 0.30).all(axis=1)
    eligible &= np.asarray([len(segment) >= 3 for segment in segments], dtype=bool)
    candidates = np.flatnonzero(eligible).tolist()
    selected = random.sample(candidates, k=round(probability * len(candidates)))
    if not selected:
        return image, labels, segments

    source_mask = np.zeros_like(image, dtype=np.uint8)
    added_labels = []
    for index in selected:
        added_labels.append(np.concatenate((labels[index, :1], mirrored_boxes[index])))
        segment = segments[index]
        segments.append(np.column_stack((width - segment[:, 0], segment[:, 1])))
        cv2.drawContours(
            source_mask,
            [segment.astype(np.int32)],
            contourIdx=-1,
            color=(1, 1, 1),
            thickness=cv2.FILLED,
        )

    labels = np.concatenate((labels, np.asarray(added_labels, dtype=np.float32)), axis=0)
    mirrored_image = cv2.flip(image, 1)
    mirrored_mask = cv2.flip(source_mask, 1).astype(bool)
    image[mirrored_mask] = mirrored_image[mirrored_mask]
    return image, labels, segments


def _resample_segment(segment: np.ndarray, count: int = 1000) -> np.ndarray:
    closed = np.concatenate((segment, segment[:1]), axis=0)
    positions = np.linspace(0, len(closed) - 1, count)
    vertices = np.arange(len(closed))
    x = np.interp(positions, vertices, closed[:, 0])
    y = np.interp(positions, vertices, closed[:, 1])
    return np.column_stack((x, y))


def _segment_box(segment: np.ndarray, width: int, height: int) -> np.ndarray:
    inside = (segment[:, 0] >= 0) & (segment[:, 1] >= 0) & (segment[:, 0] <= width) & (segment[:, 1] <= height)
    if not inside.any():
        return np.zeros(4, dtype=np.float64)
    points = segment[inside]
    return np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])


def _box_candidates(
    before: np.ndarray,
    after: np.ndarray,
    area_threshold: np.ndarray,
    scale: float,
) -> np.ndarray:
    before_width = (before[:, 2] - before[:, 0]) * scale
    before_height = (before[:, 3] - before[:, 1]) * scale
    after_width = after[:, 2] - after[:, 0]
    after_height = after[:, 3] - after[:, 1]
    aspect = np.maximum(
        after_width / (after_height + 1e-16),
        after_height / (after_width + 1e-16),
    )
    retained_area = (after_width * after_height) / (before_width * before_height + 1e-16)
    return (after_width > 2) & (after_height > 2) & (retained_area > area_threshold) & (aspect < 100)


def _random_perspective(
    image: np.ndarray,
    labels: np.ndarray,
    segments: Sequence[np.ndarray],
    degrees: float,
    translate: float,
    scale_gain: float,
    shear: float,
    perspective: float,
    border: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the recipe's combined perspective/affine matrix and filter boxes."""
    output_height = image.shape[0] + border[0] * 2
    output_width = image.shape[1] + border[1] * 2
    if output_height <= 0 or output_width <= 0:
        raise ValueError("perspective border produces an empty image")

    center = np.eye(3)
    center[0, 2] = -image.shape[1] / 2
    center[1, 2] = -image.shape[0] / 2

    project = np.eye(3)
    project[2, 0] = random.uniform(-perspective, perspective)
    project[2, 1] = random.uniform(-perspective, perspective)

    rotation = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    sampled_scale = random.uniform(1 - scale_gain, 1 + scale_gain)
    rotation[:2] = cv2.getRotationMatrix2D((0, 0), angle, sampled_scale)

    skew = np.eye(3)
    skew[0, 1] = math.tan(math.radians(random.uniform(-shear, shear)))
    skew[1, 0] = math.tan(math.radians(random.uniform(-shear, shear)))

    shift = np.eye(3)
    shift[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * output_width
    shift[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * output_height

    matrix = shift @ skew @ rotation @ project @ center
    if perspective:
        image = cv2.warpPerspective(
            image,
            matrix,
            dsize=(output_width, output_height),
            borderValue=(114, 114, 114),
        )
    else:
        image = cv2.warpAffine(
            image,
            matrix[:2],
            dsize=(output_width, output_height),
            borderValue=(114, 114, 114),
        )

    count = len(labels)
    if count == 0:
        return image, labels

    if not segments:
        segments = [np.zeros((0, 2), dtype=np.float32) for _ in range(count)]
    elif len(segments) != count:
        raise ValueError("pixel segments must be empty or aligned with labels")
    transformed = np.zeros((count, 4), dtype=np.float64)
    has_segment = np.asarray([len(segment) >= 3 for segment in segments], dtype=bool)
    for index in range(count):
        if has_segment[index]:
            points = _resample_segment(np.asarray(segments[index]))
        else:
            x1, y1, x2, y2 = labels[index, 1:5]
            points = np.array(((x1, y1), (x2, y2), (x1, y2), (x2, y1)))

        homogeneous = np.ones((len(points), 3), dtype=np.float64)
        homogeneous[:, :2] = points
        homogeneous = homogeneous @ matrix.T
        if perspective:
            points = homogeneous[:, :2] / homogeneous[:, 2:3]
        else:
            points = homogeneous[:, :2]

        if has_segment[index]:
            transformed[index] = _segment_box(points, output_width, output_height)
        else:
            transformed[index] = np.array(
                [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]
            )
            transformed[index, [0, 2]] = transformed[index, [0, 2]].clip(0, output_width)
            transformed[index, [1, 3]] = transformed[index, [1, 3]].clip(0, output_height)

    area_threshold = np.where(has_segment, 0.01, 0.10)
    keep = _box_candidates(labels[:, 1:5], transformed, area_threshold, sampled_scale)
    labels = labels[keep].copy()
    labels[:, 1:5] = transformed[keep]
    return image, labels


def _augment_hsv(image: np.ndarray, hue_gain: float, saturation_gain: float, value_gain: float) -> None:
    if not (hue_gain or saturation_gain or value_gain):
        return
    gains = np.random.uniform(-1, 1, 3) * np.array([hue_gain, saturation_gain, value_gain]) + 1
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    values = np.arange(256, dtype=gains.dtype)
    hue_lut = ((values * gains[0]) % 180).astype(np.uint8)
    saturation_lut = np.clip(values * gains[1], 0, 255).astype(np.uint8)
    value_lut = np.clip(values * gains[2], 0, 255).astype(np.uint8)
    augmented = cv2.merge(
        (
            cv2.LUT(hue, hue_lut),
            cv2.LUT(saturation, saturation_lut),
            cv2.LUT(value, value_lut),
        )
    )
    cv2.cvtColor(augmented, cv2.COLOR_HSV2RGB, dst=image)


class YOLOv9Augmentation:
    """Apply the high-augmentation YOLOv9 training recipe.

    ``get_sample`` is required when mosaic is selected.  Every callback call
    returns ``(RGB image, normalized xyxy boxes, aligned normalized segments)``.
    Mosaic remains eligible for the whole training run; this class deliberately
    has no late-epoch mosaic shutdown state.
    """

    def __init__(self, image_size: object = 640, **hyp: object) -> None:
        if isinstance(image_size, int):
            size = image_size
        elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            if image_size[0] != image_size[1]:
                raise ValueError("YOLOv9Augmentation only supports fixed square image sizes")
            size = image_size[0]
        else:
            raise ValueError("image_size must be a fixed integer or an equal (width, height) pair")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0 or size % 2:
            raise ValueError("image_size must be a positive even integer")
        self.image_size = size

        settings = dict(_DEFAULT_HYP)
        settings.update(hyp)
        for name in ("mosaic", "mixup", "copy_paste", "flipud", "fliplr"):
            settings[name] = _probability(settings[name], name)
        for name in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective"):
            settings[name] = float(settings[name])
        self.hyp = settings
        self._albumentations = self._make_albumentations(bool(settings["albumentations"]), self.image_size)

    @staticmethod
    def _make_albumentations(enabled: bool, image_size: int):
        if not enabled:
            return None
        try:
            import albumentations as A
        except ImportError as error:
            raise ImportError("albumentations=True requires the optional dependency albumentations==1.3.1") from error
        if A.__version__ != "1.3.1":
            raise RuntimeError("YOLOv9 Albumentations parity requires version 1.3.1, found {}".format(A.__version__))
        transforms = [
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.11),
                p=0.0,
            ),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0),
        ]
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    def _prepare_sample(self, sample: Sample) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        if not isinstance(sample, (tuple, list)) or len(sample) != 3:
            raise ValueError("a sample must be an (image, boxes, segments) triple")
        image = _validate_image(sample[0])
        boxes = _as_boxes(sample[1])
        segments = _as_segments(sample[2], len(boxes))
        return image, boxes, segments

    def _mosaic(
        self,
        first: Sample,
        get_sample: SampleGetter,
    ) -> Tuple[np.ndarray, np.ndarray]:
        size = self.image_size
        center_y = int(random.uniform(size // 2, size + size // 2))
        center_x = int(random.uniform(size // 2, size + size // 2))
        extra_samples = get_sample(3)
        if not isinstance(extra_samples, (tuple, list)) or len(extra_samples) != 3:
            raise ValueError("get_sample(3) must return a list of three sample triples")
        samples = [self._prepare_sample(first)]
        samples.extend(self._prepare_sample(sample) for sample in extra_samples)
        random.shuffle(samples)

        canvas = np.full((size * 2, size * 2, 3), 114, dtype=np.uint8)
        all_labels = []
        all_segments = []
        for position, (image, labels, segments) in enumerate(samples):
            image = _resize_longest(image, size)
            height, width = image.shape[:2]
            if position == 0:
                dst = (max(center_x - width, 0), max(center_y - height, 0), center_x, center_y)
                src = (width - (dst[2] - dst[0]), height - (dst[3] - dst[1]), width, height)
            elif position == 1:
                dst = (center_x, max(center_y - height, 0), min(center_x + width, 2 * size), center_y)
                src = (0, height - (dst[3] - dst[1]), min(width, dst[2] - dst[0]), height)
            elif position == 2:
                dst = (max(center_x - width, 0), center_y, center_x, min(2 * size, center_y + height))
                src = (width - (dst[2] - dst[0]), 0, width, min(dst[3] - dst[1], height))
            else:
                dst = (center_x, center_y, min(center_x + width, 2 * size), min(center_y + height, 2 * size))
                src = (0, 0, min(width, dst[2] - dst[0]), min(height, dst[3] - dst[1]))

            x1a, y1a, x2a, y2a = dst
            x1b, y1b, x2b, y2b = src
            canvas[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            offset_x, offset_y = x1a - x1b, y1a - y1b

            if len(labels):
                pixel_labels = labels.copy()
                pixel_labels[:, [1, 3]] = labels[:, [1, 3]] * width + offset_x
                pixel_labels[:, [2, 4]] = labels[:, [2, 4]] * height + offset_y
                all_labels.append(pixel_labels)
                all_segments.extend(
                    (
                        np.column_stack((segment[:, 0] * width + offset_x, segment[:, 1] * height + offset_y))
                        if len(segment)
                        else np.zeros((0, 2), dtype=np.float32)
                    )
                    for segment in segments
                )

        if all_labels:
            labels = np.concatenate(all_labels, axis=0)
            labels[:, 1:] = labels[:, 1:].clip(0, 2 * size)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)
        for segment in all_segments:
            np.clip(segment, 0, 2 * size, out=segment)

        canvas, labels, all_segments = _copy_paste(canvas, labels, all_segments, self.hyp["copy_paste"])
        return _random_perspective(
            canvas,
            labels,
            all_segments,
            self.hyp["degrees"],
            self.hyp["translate"],
            self.hyp["scale"],
            self.hyp["shear"],
            self.hyp["perspective"],
            border=(-size // 2, -size // 2),
        )

    def _single(self, sample: Sample) -> Tuple[np.ndarray, np.ndarray]:
        image, labels, _ = self._prepare_sample(sample)
        image = _resize_longest(image, self.image_size)
        height, width = image.shape[:2]
        pad_x = (self.image_size - width) / 2
        pad_y = (self.image_size - height) / 2
        left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
        top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if len(labels):
            labels[:, [1, 3]] = labels[:, [1, 3]] * width + pad_x
            labels[:, [2, 4]] = labels[:, [2, 4]] * height + pad_y
        return _random_perspective(
            image,
            labels,
            [],
            self.hyp["degrees"],
            self.hyp["translate"],
            self.hyp["scale"],
            self.hyp["shear"],
            self.hyp["perspective"],
        )

    def _apply_albumentations(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._albumentations is None:
            return image, labels
        # The reference loader stores BGR images and gives normalized xywh
        # labels to Albumentations. Preserve that boundary despite this module's
        # public RGB/xyxy interface.
        if random.random() >= 1.0:
            return image, labels
        bgr_image = np.ascontiguousarray(image[:, :, ::-1])
        yolo_boxes = labels[:, 1:].copy()
        if len(yolo_boxes):
            yolo_boxes[:, 0] = (labels[:, 1] + labels[:, 3]) / 2
            yolo_boxes[:, 1] = (labels[:, 2] + labels[:, 4]) / 2
            yolo_boxes[:, 2] = labels[:, 3] - labels[:, 1]
            yolo_boxes[:, 3] = labels[:, 4] - labels[:, 2]
        transformed = self._albumentations(
            image=bgr_image,
            bboxes=yolo_boxes.tolist(),
            class_labels=labels[:, 0].tolist(),
        )
        image = np.ascontiguousarray(transformed["image"][:, :, ::-1])
        if transformed["bboxes"]:
            converted = np.asarray(transformed["bboxes"], dtype=np.float32)
            boxes_xyxy = converted.copy()
            boxes_xyxy[:, 0] = converted[:, 0] - converted[:, 2] / 2
            boxes_xyxy[:, 1] = converted[:, 1] - converted[:, 3] / 2
            boxes_xyxy[:, 2] = converted[:, 0] + converted[:, 2] / 2
            boxes_xyxy[:, 3] = converted[:, 1] + converted[:, 3] / 2
            labels = np.column_stack((np.asarray(transformed["class_labels"], dtype=np.float32), boxes_xyxy)).astype(
                np.float32, copy=False
            )
        else:
            labels = np.zeros((0, 5), dtype=np.float32)
        return image, labels

    def __call__(
        self,
        image: np.ndarray,
        boxes: object = None,
        segments: object = None,
        get_sample: Optional[SampleGetter] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = self._prepare_sample((image, boxes, segments))
        use_mosaic = random.random() < self.hyp["mosaic"]
        if use_mosaic:
            if get_sample is None:
                raise ValueError("get_sample is required when mosaic augmentation is selected")
            image, labels = self._mosaic(first, get_sample)
            if random.random() < self.hyp["mixup"]:
                second = self._prepare_sample(get_sample(1))
                image2, labels2 = self._mosaic(second, get_sample)
                ratio = np.random.beta(32.0, 32.0)
                image = (image * ratio + image2 * (1 - ratio)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), axis=0)
        else:
            image, labels = self._single(first)

        height, width = image.shape[:2]
        if (height, width) != (self.image_size, self.image_size):
            raise RuntimeError("YOLOv9 geometry did not produce the configured square output")
        if len(labels):
            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, width - 1e-3)
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, height - 1e-3)
            labels[:, [1, 3]] /= width
            labels[:, [2, 4]] /= height

        image, labels = self._apply_albumentations(image, labels)
        _augment_hsv(
            image,
            self.hyp["hsv_h"],
            self.hyp["hsv_s"],
            self.hyp["hsv_v"],
        )

        if random.random() < self.hyp["flipud"]:
            image = np.flipud(image)
            if len(labels):
                labels[:, [2, 4]] = 1 - labels[:, [4, 2]]
        if random.random() < self.hyp["fliplr"]:
            image = np.fliplr(image)
            if len(labels):
                labels[:, [1, 3]] = 1 - labels[:, [3, 1]]

        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(torch.float32).div_(255.0)
        boxes_tensor = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))
        reverse = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        return image_tensor, boxes_tensor, reverse


__all__ = ["YOLOv9Augmentation"]
