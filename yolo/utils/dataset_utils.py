import json
import os
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import ListConfig

from yolo.tools.data_conversion import discretize_categories
from yolo.utils.annotation_utils import parse_coco_bbox
from yolo.utils.logger import logger


def normalize_dataset_inputs(value, phase: str) -> List[str]:
    """Accept a split name/Parquet path or a non-empty YAML/Python list of inputs."""
    inputs = [value] if isinstance(value, str) else value
    if not isinstance(inputs, (list, tuple, ListConfig)) or not inputs:
        raise ValueError(f"dataset.{phase} must be a string or a non-empty list of strings")
    if any(not isinstance(item, str) or not item.strip() for item in inputs):
        raise ValueError(f"dataset.{phase} must contain only non-empty strings")
    return list(inputs)


def locate_label_paths(dataset_path: Path, phase_name: Path) -> Tuple[Path, Path]:
    """
    Find the path to label files for a specified dataset and phase(e.g. training).

    Args:
        dataset_path (Path): The path to the root directory of the dataset.
        phase_name (Path): The name of the phase for which labels are being searched (e.g., "train", "val", "test").

    Returns:
        Tuple[Path, Path]: A tuple containing the path to the labels file and the file format ("json" or "txt").
    """
    json_labels_path = dataset_path / "annotations" / f"instances_{phase_name}.json"

    txt_labels_path = dataset_path / "labels" / phase_name

    if json_labels_path.is_file():
        return json_labels_path, "json"

    elif txt_labels_path.is_dir():
        txt_files = [f for f in os.listdir(txt_labels_path) if f.endswith(".txt")]
        if txt_files:
            return txt_labels_path, "txt"

    logger.warning("No labels found in the specified dataset path and phase name.")
    return [], None


def create_image_metadata(labels_path: str) -> Tuple[Dict[str, List], Dict[str, Dict]]:
    """
    Create a dictionary containing image information and annotations indexed by image ID.

    Args:
        labels_path (str): The path to the annotation json file.

    Returns:
        - annotations_index: A dictionary where keys are image IDs and values are lists of annotations.
        - image_info_dict: A dictionary where keys are image file names without extension and values are image information dictionaries.
    """
    with open(labels_path, "r") as file:
        labels_data = json.load(file)
        id_to_idx = discretize_categories(labels_data.get("categories", [])) if "categories" in labels_data else None
        annotations_index = organize_annotations_by_image(labels_data, id_to_idx)  # check lookup is a good name?
        image_info_dict = {Path(img["file_name"]).stem: img for img in labels_data["images"]}
        return annotations_index, image_info_dict


def organize_annotations_by_image(data: Dict[str, Any], id_to_idx: Optional[Dict[int, int]]):
    """
    Use image index to lookup every annotations
    Args:
        data (Dict[str, Any]): A dictionary containing annotation data.

    Returns:
        Dict[int, List[Dict[str, Any]]]: A dictionary where keys are image IDs and values are lists of annotations.
        Annotations with "iscrowd" set to True are excluded from the index.

    """
    annotation_lookup = {}
    for anno in data["annotations"]:
        if anno.get("iscrowd", False):
            continue
        image_id = anno["image_id"]
        if id_to_idx:
            if anno["category_id"] not in id_to_idx:
                raise ValueError(f"Unknown COCO category_id {anno['category_id']!r} at COCO annotation {anno.get('id')}")
            anno["category_id"] = id_to_idx[anno["category_id"]]
        if image_id not in annotation_lookup:
            annotation_lookup[image_id] = []
        annotation_lookup[image_id].append(anno)
    return annotation_lookup


def scale_segmentation(
    annotations: List[Dict[str, Any]], image_dimensions: Dict[str, int]
) -> Optional[List[List[float]]]:
    """
    Scale the segmentation data based on image dimensions and return a list of scaled segmentation data.

    Args:
        annotations (List[Dict[str, Any]]): A list of annotation dictionaries.
        image_dimensions (Dict[str, int]): A dictionary containing image dimensions (height and width).

    Returns:
        Optional[List[List[float]]]: A list of scaled segmentation data, where each sublist contains category_id followed by scaled (x, y) coordinates.
    """
    if annotations is None:
        return None

    seg_array_with_cat = []
    h, w = image_dimensions["height"], image_dimensions["width"]
    if not np.isfinite([w, h]).all() or w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions for COCO image {image_dimensions.get('id')}")
    for anno in annotations:
        category_id = anno["category_id"]
        # Detection uses the authoritative bbox even if segmentation is empty,
        # RLE, or has a different extent. This legacy API emits rectangle points.
        if "bbox" in anno:
            box = parse_coco_bbox(anno, w, h)
            if box is not None:
                cls, x1, y1, x2, y2 = box
                seg_array_with_cat.append([cls, x1, y1, x2, y1, x2, y2, x1, y2])
            continue
        segmentation = anno.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            continue
        seg_list = list(chain.from_iterable(segmentation)) if isinstance(segmentation[0], list) else segmentation

        scaled_seg_data = (
            np.array(seg_list).reshape(-1, 2) / [w, h]
        ).tolist()  # make the list group in x, y pairs and scaled with image width, height
        scaled_flat_seg_data = [category_id] + list(chain(*scaled_seg_data))  # flatten the scaled_seg_data list
        seg_array_with_cat.append(scaled_flat_seg_data)

    return seg_array_with_cat


def tensorlize(data):
    try:
        img_paths, bboxes, img_ratios = zip(*data)
    except ValueError as e:
        logger.error(
            ":rotating_light: This may be caused by using old cache or another version of YOLO's cache.\n"
            ":rotating_light: Please clean the cache and try running again."
        )
        raise e
    max_box = max(bbox.size(0) for bbox in bboxes)
    padded_bbox_list = []
    for bbox in bboxes:
        padding = torch.full((max_box, 5), -1, dtype=torch.float32)
        padding[: bbox.size(0)] = bbox
        padded_bbox_list.append(padding)
    bboxes = np.stack(padded_bbox_list)
    img_paths = np.array(img_paths)
    img_ratios = np.array(img_ratios)
    return img_paths, bboxes, img_ratios
