"""Official COCO human-keypoint evaluation in original image coordinates."""

import contextlib
import io
import math

import torch
from pycocotools.cocoeval import COCOeval

from yolo.utils.coco_eval import CocoJsonEvaluator

POSE_METRIC_NAMES = (
    "map",
    "map_50",
    "map_75",
    "map_medium",
    "map_large",
    "mar_20",
    "mar_20_50",
    "mar_20_75",
    "mar_20_medium",
    "mar_20_large",
)


class CocoPoseEvaluator(CocoJsonEvaluator):
    """Evaluate rows ``[class, x1, y1, x2, y2, score, (x,y,confidence)*17]``.

    Uses pycocotools' official 17 human-joint OKS sigmas, maxDets=20,
    authoritative GT visibility/area/crowd, and only observed image IDs.
    Detection score determines instance ranking. Keypoint confidences are
    retained in results, but COCO OKS masks joints using GT visibility.
    ``map`` is pose AP and is also the generic checkpoint-selection metric.
    """

    metric_names = POSE_METRIC_NAMES

    def __init__(self, annotation_path, image_root=None, num_keypoints=17):
        if num_keypoints != 17:
            raise ValueError("Official COCO pose evaluation requires the 17-keypoint human layout")
        self.num_keypoints = 17
        super().__init__(annotation_path, image_root)
        for category in self.coco_gt.dataset.get("categories", []):
            if "keypoints" in category and len(category["keypoints"]) != self.num_keypoints:
                raise ValueError("Official COCO pose evaluation requires 17 keypoints per category")
        for annotation in self.coco_gt.dataset.get("annotations", []):
            keypoints = annotation.get("keypoints")
            if not isinstance(keypoints, (list, tuple)) or len(keypoints) != 3 * self.num_keypoints:
                raise ValueError(f"COCO annotation {annotation.get('id')} must contain 17 keypoint triplets")
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in keypoints):
                raise ValueError(f"COCO annotation {annotation.get('id')} has non-finite keypoints")
            if any(value not in (0, 1, 2) for value in keypoints[2::3]):
                raise ValueError(f"COCO annotation {annotation.get('id')} has invalid joint visibility")
            annotation.setdefault("num_keypoints", sum(value > 0 for value in keypoints[2::3]))

    def _convert_predictions(self, prediction, image_id, target_width, target_height):
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("each prediction must be a torch.Tensor")
        expected_width = 6 + 3 * self.num_keypoints
        if prediction.ndim != 2 or prediction.shape[1] != expected_width:
            raise ValueError(f"each pose prediction tensor must have shape [N, {expected_width}]")
        # Reuse box/class validation, category mapping and exact letterbox
        # inversion. Pose coordinates follow the same integer resize geometry.
        detections = super()._convert_predictions(prediction[:, :6], image_id, target_width, target_height)
        keypoints = prediction[:, 6:].detach().to(device="cpu", dtype=torch.float64).reshape(-1, 17, 3)
        if not torch.isfinite(keypoints).all():
            raise ValueError("pose prediction contains a non-finite keypoint value")
        image = self._images[image_id]
        original_width, original_height = int(image["width"]), int(image["height"])
        scale = min(target_width / original_width, target_height / original_height)
        resized_width, resized_height = int(original_width * scale), int(original_height * scale)
        pad_left, pad_top = (target_width - resized_width) // 2, (target_height - resized_height) // 2
        # Do not clip predictions to image boundaries: that changes OKS errors.
        keypoints = keypoints.clone()
        keypoints[..., 0] = (keypoints[..., 0] - pad_left) / (resized_width / original_width)
        keypoints[..., 1] = (keypoints[..., 1] - pad_top) / (resized_height / original_height)
        for detection, joints in zip(detections, keypoints):
            detection["keypoints"] = joints.flatten().tolist()
        return detections

    def _evaluate(self, records):
        if not records:
            raise ValueError("no images have been added to the COCO evaluator")
        detections = [detection for record in records.values() for detection in record]
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self.coco_gt.loadRes(detections) if detections else self._empty_results()
            evaluator = COCOeval(self.coco_gt, coco_dt, "keypoints")
            evaluator.params.imgIds = list(records.keys())
            evaluator.params.catIds = list(self._category_ids)
            # COCOeval keypoint defaults supply the official human OKS sigmas
            # and maxDets=[20]; keep these intact for standard comparability.
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        return {name: float(value) for name, value in zip(self.metric_names, evaluator.stats.tolist())}
