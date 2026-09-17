"""Shared discovery and parsing of bounding-box Parquet annotations."""

from pathlib import Path
from typing import Optional

import numpy as np

from yolo.utils.annotation_utils import parse_yolo_label, validate_class


PARQUET_COLUMNS = ("image", "conf", "id_class", "box_cx", "box_cy", "box_w", "box_h")


def parquet_split_name(source: str) -> str:
    """Infer the image split from an explicit Parquet filename."""
    if Path(source).suffix.lower() != ".parquet":
        return source
    name = Path(source).stem
    return name.removeprefix("instances_")


def resolve_annotation_image_path(dataset_path: Path, split: str, image: str) -> Path:
    """Match the COCO JSON loader's absolute and relative filename lookup."""
    source = Path(image)
    if source.is_absolute():
        return source
    candidates = (
        dataset_path / "images" / split / source,
        dataset_path / "images" / source,
        dataset_path / source,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def resolve_parquet_annotation(dataset_path: Path, source: str) -> Optional[Path]:
    """Resolve an explicit Parquet path or the conventional split annotation.

    Existing TXT/JSON split precedence is unchanged. An explicit .parquet
    request is returned even when missing, so it fails instead of falling back.
    """
    path = Path(source)
    if path.suffix.lower() == ".parquet":
        return path if path.is_absolute() else dataset_path / path
    if (dataset_path / f"{source}.txt").is_file():
        return None
    annotations = dataset_path / "annotations"
    if (annotations / f"instances_{source}.json").is_file():
        return None
    path = annotations / f"instances_{source}.parquet"
    return path if path.is_file() else None


def load_parquet_annotations(annotation_path: Path, dataset_path: Path, class_num=None, split=None):
    """Group one-box-per-row annotations into sorted image/normalized-xyxy pairs.

    Confidence is metadata, not a target coordinate, sample weight, or
    filtering threshold. Image paths follow the COCO JSON lookup convention.
    id_class is the zero-based class_list/model index; no remapping is applied.
    """
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Parquet annotation file does not exist: {annotation_path}")
    # Keep existing TXT/JSON consumers usable before optional imports occur.
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("Parquet annotations require pandas and pyarrow; install requirements.txt") from error

    columns = pq.read_schema(annotation_path).names
    missing = [column for column in PARQUET_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Missing Parquet annotation columns {missing} in {annotation_path}")
    frame = pd.read_parquet(annotation_path, columns=list(PARQUET_COLUMNS), engine="pyarrow")
    if frame.empty:
        raise ValueError(f"No annotation rows found in {annotation_path}")

    grouped = {}
    image_paths = {}
    for row_number, row in enumerate(frame.itertuples(index=False, name=None), start=1):
        image, _confidence, class_id, *coordinates = row
        location = f"{annotation_path}:row {row_number}"
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"image must be a non-empty path string at {location}")
        try:
            class_id = float(class_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"id_class must be a non-negative integer at {location}") from error
        # Validate before float32 conversion, which could round fractional IDs.
        validate_class(class_id, class_num, location)
        box, _ = parse_yolo_label([class_id, *coordinates], location, class_num)
        if image not in image_paths:
            image_paths[image] = resolve_annotation_image_path(
                dataset_path, split or parquet_split_name(str(annotation_path)), image
            ).resolve()
        image_path = image_paths[image]
        if image_path not in grouped:
            if not image_path.is_file():
                raise FileNotFoundError(f"Image at {location} does not exist: {image_path}")
            grouped[image_path] = []
        grouped[image_path].append(box)

    return [
        (path, np.asarray(grouped[path], dtype=np.float32).reshape(-1, 5))
        for path in sorted(grouped, key=str)
    ]
