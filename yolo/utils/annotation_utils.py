"""Shared detection label parsing for training and validation loaders."""

from collections.abc import Sequence

import numpy as np

from yolo.utils.logger import logger


def validate_class(cls, class_num, location):
    if not np.isfinite(cls) or cls < 0 or not float(cls).is_integer():
        raise ValueError(f"Class must be a non-negative integer at {location}")
    if class_num is not None and cls >= int(class_num):
        raise ValueError(f"Class {int(cls)} exceeds class_num={class_num} at {location}")


def polygon_area(points):
    # Avoid float32 cancellation for small valid boxes near the image edge.
    points = np.asarray(points, dtype=np.float64)
    points = points - points[0]
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def parse_yolo_label(values, location, class_num=None):
    """Return normalized xyxy and an optional real polygon from one TXT row.

    Detection rows are class + center xywh; longer rows are class + xy points.
    Validate the source before any clipping. Derived detection corners may lie
    outside the image, as in the original YOLOv9 parser. The caller clips them
    either before legacy transforms or during YOLOv9 geometric augmentation.
    """
    try:
        values = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric label at {location}") from error
    if values.ndim != 1 or not (values.size == 5 or (values.size >= 7 and values.size % 2 == 1)):
        raise ValueError(f"Expected 5 detection values or class plus at least 3 xy points at {location}")
    cls = float(values[0])
    validate_class(cls, class_num, location)
    coordinates = values[1:]
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Coordinates must be finite at {location}")
    if (coordinates < 0).any() or (coordinates > 1).any():
        raise ValueError(f"Coordinates must be normalized to [0, 1] at {location}")
    if values.size == 5:
        xc, yc, width, height = coordinates.tolist()
        if width <= 0 or height <= 0:
            raise ValueError(f"Box width and height must be positive at {location}")
        box = [cls, xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2]
        segment = np.zeros((0, 2), dtype=np.float32)
    else:
        segment = coordinates.reshape(-1, 2).copy()
        if abs(polygon_area(segment)) <= 0:
            raise ValueError(f"Segmentation polygon has zero area at {location}")
        minimum, maximum = segment.min(axis=0), segment.max(axis=0)
        box = [cls, *minimum, *maximum]
    return box, segment


def parse_coco_bbox(annotation, image_width, image_height, category_map=None, class_num=None):
    """Use COCO's pixel xywh bbox, independent of optional segmentation.

    Clip partial boxes to the image; skip malformed, nonfinite, nonpositive,
    or wholly outside boxes. Invalid classes are configuration errors.
    """
    location = f"COCO annotation {annotation.get('id', '<unknown>')}"
    if not np.isfinite([image_width, image_height]).all() or image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image dimensions at {location}")
    bbox = annotation.get("bbox")
    try:
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
            raise ValueError("bbox must be a four-element sequence")
        coordinates = np.asarray(bbox, dtype=np.float64)
        if coordinates.shape != (4,) or not np.isfinite(coordinates).all() or (coordinates[2:] <= 0).any():
            raise ValueError("bbox must contain finite xywh with positive width and height")
    except (TypeError, ValueError) as error:
        logger.warning(f"Skipping invalid bbox at {location}: {error}")
        return None

    category_id = annotation.get("category_id")
    if category_map is not None:
        if category_id not in category_map:
            raise ValueError(f"Unknown COCO category_id {category_id!r} at {location}")
        cls = category_map[category_id]
    else:
        cls = category_id
    try:
        cls = float(cls)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Class must be a non-negative integer at {location}") from error
    validate_class(cls, class_num, location)

    x, y, width, height = coordinates
    x1, y1 = np.clip([x / image_width, y / image_height], 0.0, 1.0)
    x2, y2 = np.clip([(x + width) / image_width, (y + height) / image_height], 0.0, 1.0)
    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Skipping bbox outside the image at {location}")
        return None
    return [cls, float(x1), float(y1), float(x2), float(y2)]
