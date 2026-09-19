"""Official COCO JSON based bounding-box evaluation.

The evaluator deliberately uses the annotations from the supplied COCO JSON as
ground truth.  Predictions are retained in COCO result format and can be read
through :attr:`predictions` (or :meth:`export`) without writing temporary files.
"""

import contextlib
import copy
import io
import json
import math
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor

METRIC_NAMES = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)


def _normalise_path(path: Union[str, os.PathLike]) -> str:
    """Return a platform-independent spelling suitable for lookup only."""
    try:
        value = os.fspath(path)
    except TypeError as exc:
        raise TypeError("image paths must be strings or path-like objects") from exc
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    # COCO file_name uses forward slashes even when a dataset is consumed on a
    # different platform.  normpath also removes harmless './' components.
    return os.path.normpath(value.replace("\\", "/")).replace("\\", "/")


class CocoJsonEvaluator:
    """Accumulate detections and evaluate them against an official COCO JSON.

    Args:
        annotation_path: Path to a COCO instances annotation JSON file, or a
            list of files with identical category IDs/names to evaluate jointly.
        image_root: Optional directory against which annotation ``file_name``
            entries and incoming image paths are resolved. For multiple JSON
            files, supply one root per file; IDs are remapped in memory.

    ``update`` expects one tensor per image with rows ordered as
    ``[contiguous_class, x1, y1, x2, y2, score]``.  Coordinates are in the
    padded model input described by the shared ``image_size=[width, height]``.
    Contiguous classes are mapped to category IDs sorted in the same way as
    :func:`yolo.tools.data_conversion.discretize_categories`.
    """

    def __init__(
        self,
        annotation_path: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        image_root: Optional[Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]] = None,
    ):
        if isinstance(annotation_path, (list, tuple)):
            self.annotation_path = [Path(path) for path in annotation_path]
            if not isinstance(image_root, (list, tuple)) or len(image_root) != len(self.annotation_path):
                raise ValueError("Multiple COCO annotations require one image_root per input")
            # Use absolute filenames and fresh IDs so independent exports may
            # safely reuse image basenames and annotation/image IDs.
            self.image_root = Path.cwd()
            merged = {"images": [], "annotations": [], "categories": []}
            expected_categories = None
            for path, root in zip(self.annotation_path, image_root):
                with path.open(encoding="utf-8") as source:
                    data = json.load(source)
                categories = sorted(data.get("categories", []), key=lambda category: category["id"])
                signature = [(category["id"], category.get("name")) for category in categories]
                if expected_categories is not None and signature != expected_categories:
                    raise ValueError("Multiple COCO inputs must use the same category IDs and names")
                expected_categories = signature
                merged["categories"] = categories
                image_ids = {}
                for image in data.get("images", []):
                    old_id = image["id"]
                    if old_id in image_ids:
                        raise ValueError(f"COCO annotation contains duplicate image ID {old_id!r}: {path}")
                    image_ids[old_id] = len(merged["images"]) + 1
                    merged["images"].append(
                        dict(image, id=image_ids[old_id], file_name=str((Path(root) / image["file_name"]).resolve()))
                    )
                for annotation in data.get("annotations", []):
                    merged["annotations"].append(
                        dict(annotation, id=len(merged["annotations"]) + 1, image_id=image_ids[annotation["image_id"]])
                    )
            with contextlib.redirect_stdout(io.StringIO()):
                self._coco_gt = COCO()
                self._coco_gt.dataset = merged
                self._coco_gt.createIndex()
        else:
            self.annotation_path = Path(annotation_path)
            if not self.annotation_path.is_file():
                raise FileNotFoundError("COCO annotation file does not exist: {}".format(self.annotation_path))

            self.image_root = Path(image_root).resolve() if image_root is not None else None
            with contextlib.redirect_stdout(io.StringIO()):
                self._coco_gt = COCO(str(self.annotation_path))

        # Minimal detection-only exports often omit these optional fields.
        # Keep supplied COCO area/crowd values; derive only missing metadata in
        # memory, without rewriting the user's annotations or inventing masks.
        for annotation in self._coco_gt.dataset.get("annotations", []):
            annotation.setdefault("iscrowd", 0)
            if "area" not in annotation:
                bbox = annotation.get("bbox")
                if (
                    not isinstance(bbox, (list, tuple))
                    or len(bbox) != 4
                    or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox)
                    or bbox[2] <= 0
                    or bbox[3] <= 0
                ):
                    raise ValueError(f"Cannot derive area from bbox for COCO annotation {annotation.get('id')}")
                annotation["area"] = bbox[2] * bbox[3]

        categories = self._coco_gt.dataset.get("categories", [])
        category_ids = [category["id"] for category in sorted(categories, key=lambda category: category["id"])]
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("COCO annotation contains duplicate category IDs")
        self._category_ids = tuple(category_ids)

        self._images = {}
        relative_lookup = defaultdict(list)
        absolute_lookup = defaultdict(list)
        basename_lookup = defaultdict(list)
        for image in self._coco_gt.dataset.get("images", []):
            image_id = image.get("id")
            if image_id in self._images:
                raise ValueError("COCO annotation contains duplicate image ID {!r}".format(image_id))
            if "file_name" not in image or "width" not in image or "height" not in image:
                raise ValueError("each COCO image needs id, file_name, width, and height")
            if image["width"] <= 0 or image["height"] <= 0:
                raise ValueError("COCO image dimensions must be positive for image ID {!r}".format(image_id))

            self._images[image_id] = image
            relative_name = _normalise_path(image["file_name"])
            relative_lookup[relative_name].append(image_id)
            basename_lookup[os.path.basename(relative_name)].append(image_id)
            if self.image_root is not None:
                absolute_name = _normalise_path((self.image_root / relative_name).resolve())
                absolute_lookup[absolute_name].append(image_id)

        self._relative_lookup = dict(relative_lookup)
        self._absolute_lookup = dict(absolute_lookup)
        self._basename_lookup = dict(basename_lookup)
        self.reset()

    @property
    def coco_gt(self) -> COCO:
        """Expose the loaded annotation object for category metadata checks."""
        return self._coco_gt

    @property
    def predictions(self) -> List[Dict]:
        """Return a copy of accumulated detections in COCO result format."""
        return copy.deepcopy([detection for record in self._records.values() for detection in record])

    @property
    def image_ids(self) -> List:
        """Return all accumulated image IDs, including images with no detections."""
        return list(self._records.keys())

    def export(self) -> List[Dict]:
        """Return COCO-format detections without writing them to disk."""
        return self.predictions

    def reset(self) -> None:
        """Clear accumulated predictions while retaining the loaded annotation."""
        self._records = OrderedDict()

    def _resolve_image_id(self, image_path: Union[str, os.PathLike]):
        incoming = _normalise_path(image_path)

        candidates = self._relative_lookup.get(incoming, [])
        if not candidates and self.image_root is not None:
            path = Path(os.fspath(image_path))
            # A relative caller path may be relative either to the process cwd
            # (for example ``data/coco/images/val/a.jpg``) or to image_root
            # (``a.jpg``).  Try both spellings before the basename fallback.
            absolute_paths = [path.resolve()]
            if not path.is_absolute():
                absolute_paths.append((self.image_root / path).resolve())
            for absolute_path in absolute_paths:
                candidates = self._absolute_lookup.get(_normalise_path(absolute_path), [])
                if candidates:
                    break
        if not candidates:
            candidates = self._basename_lookup.get(os.path.basename(incoming), [])

        unique_candidates = list(dict.fromkeys(candidates))
        if not unique_candidates:
            raise ValueError("image path is not present in the COCO annotation: {}".format(image_path))
        if len(unique_candidates) != 1:
            raise ValueError("image path is ambiguous in the COCO annotation: {}".format(image_path))
        return unique_candidates[0]

    @staticmethod
    def _validate_image_size(image_size: Sequence[int]) -> Tuple[int, int]:
        if isinstance(image_size, Tensor):
            image_size = image_size.detach().cpu().tolist()
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise ValueError("image_size must be [width, height]")
        width, height = image_size
        if isinstance(width, bool) or isinstance(height, bool):
            raise ValueError("image_size dimensions must be positive integers")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError("image_size dimensions must be positive integers")
        return width, height

    def _convert_predictions(
        self,
        prediction: Tensor,
        image_id,
        target_width: int,
        target_height: int,
    ) -> List[Dict]:
        if not isinstance(prediction, Tensor):
            raise TypeError("each prediction must be a torch.Tensor")
        if prediction.ndim != 2 or prediction.shape[1] != 6:
            raise ValueError("each prediction tensor must have shape [N, 6]")

        image = self._images[image_id]
        original_width = int(image["width"])
        original_height = int(image["height"])
        scale = min(target_width / original_width, target_height / original_height)
        resized_width = int(original_width * scale)
        resized_height = int(original_height * scale)
        if resized_width <= 0 or resized_height <= 0:
            raise ValueError("model image_size is too small for image ID {!r}".format(image_id))
        pad_left = (target_width - resized_width) // 2
        pad_top = (target_height - resized_height) // 2
        gain_x = resized_width / original_width
        gain_y = resized_height / original_height

        rows = prediction.detach().to(device="cpu", dtype=torch.float64).tolist()
        detections = []
        for row_index, (class_value, x1, y1, x2, y2, score) in enumerate(rows):
            if not all(math.isfinite(value) for value in (class_value, x1, y1, x2, y2, score)):
                raise ValueError("prediction row {} contains a non-finite value".format(row_index))
            if not float(class_value).is_integer():
                raise ValueError("prediction class must be an integer index, got {!r}".format(class_value))
            class_index = int(class_value)
            if class_index < 0 or class_index >= len(self._category_ids):
                raise ValueError("prediction class index {} is out of range".format(class_index))
            if x2 < x1 or y2 < y1:
                raise ValueError("prediction row {} has inverted box coordinates".format(row_index))

            original_x1 = min(max((x1 - pad_left) / gain_x, 0.0), float(original_width))
            original_y1 = min(max((y1 - pad_top) / gain_y, 0.0), float(original_height))
            original_x2 = min(max((x2 - pad_left) / gain_x, 0.0), float(original_width))
            original_y2 = min(max((y2 - pad_top) / gain_y, 0.0), float(original_height))
            box_width = original_x2 - original_x1
            box_height = original_y2 - original_y1
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": self._category_ids[class_index],
                    "bbox": [original_x1, original_y1, box_width, box_height],
                    "score": float(score),
                }
            )
        return detections

    def update(
        self,
        predictions: Sequence[Tensor],
        image_paths: Sequence[Union[str, os.PathLike]],
        image_size: Sequence[int],
    ) -> None:
        """Add one batch of padded-input detections to the evaluation state."""
        target_width, target_height = self._validate_image_size(image_size)
        if len(predictions) != len(image_paths):
            raise ValueError("predictions and image_paths must have the same length")

        for prediction, image_path in zip(predictions, image_paths):
            image_id = self._resolve_image_id(image_path)
            detections = self._convert_predictions(prediction, image_id, target_width, target_height)
            if image_id in self._records:
                if self._records[image_id] != detections:
                    raise ValueError("conflicting predictions were supplied for image ID {!r}".format(image_id))
                continue
            self._records[image_id] = detections

    def _empty_results(self) -> COCO:
        """Build a valid empty result set without COCO.loadRes([])."""
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": copy.deepcopy(self._coco_gt.dataset.get("images", [])),
            "categories": copy.deepcopy(self._coco_gt.dataset.get("categories", [])),
            "annotations": [],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt.createIndex()
        return coco_dt

    def _evaluate(self, records: "OrderedDict") -> Dict[str, float]:
        if not records:
            raise ValueError("no images have been added to the COCO evaluator")
        detections = [detection for record in records.values() for detection in record]
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self._coco_gt.loadRes(detections) if detections else self._empty_results()
            coco_eval = COCOeval(self._coco_gt, coco_dt, "bbox")
            coco_eval.params.imgIds = list(records.keys())
            coco_eval.params.catIds = list(self._category_ids)
            coco_eval.params.maxDets = [1, 10, 100]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        return {name: float(value) for name, value in zip(METRIC_NAMES, coco_eval.stats.tolist())}

    @staticmethod
    def _distributed() -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _gather_records(self):
        local_records = [(image_id, copy.deepcopy(detections)) for image_id, detections in self._records.items()]
        if not self._distributed():
            return OrderedDict(local_records)

        # Only the evaluating rank needs the global predictions. Replicating
        # millions of COCO detection dictionaries on every rank wastes RAM.
        gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(local_records, object_gather_list=gathered, dst=0)
        if dist.get_rank() != 0:
            return None

        # Rank order is stable. DistributedSampler padding can repeat images;
        # retaining the lowest-rank occurrence makes the choice deterministic.
        merged = OrderedDict()
        for rank_records in gathered:
            for image_id, detections in rank_records:
                if image_id not in merged:
                    merged[image_id] = detections
        return merged

    def compute(self) -> Dict[str, Tensor]:
        """Run COCOeval over exactly the images observed by all ranks.

        In distributed runs rank zero evaluates the rank-ordered, de-duplicated
        records and broadcasts either the compact result or the error, ensuring
        that peer ranks do not wait indefinitely after an evaluation failure.
        """
        distributed = self._distributed()
        records = self._gather_records()
        if not distributed:
            metrics = self._evaluate(records)
            result = {name: torch.tensor(metrics[name], dtype=torch.float64) for name in METRIC_NAMES}
            result["classes"] = torch.arange(len(self._category_ids), dtype=torch.int64)
            return result

        rank = dist.get_rank()
        packet = None
        if rank == 0:
            try:
                packet = {"ok": True, "metrics": self._evaluate(records)}
            except Exception as exc:  # broadcast failure before re-raising on every rank
                packet = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}

        packets = [packet]
        dist.broadcast_object_list(packets, src=0)
        packet = packets[0]

        if not packet["ok"]:
            raise RuntimeError("COCO evaluation failed ({}): {}".format(packet["error_type"], packet["error"]))

        result = {name: torch.tensor(packet["metrics"][name], dtype=torch.float64) for name in METRIC_NAMES}
        result["classes"] = torch.arange(len(self._category_ids), dtype=torch.int64)
        return result
