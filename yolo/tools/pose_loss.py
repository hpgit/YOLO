"""Grid-relative pose distributions trained on every detection TAL positive.

COCO v=1 (occluded) and v=2 (visible) both supervise coordinates and receive
confidence target 1. The confidence predicts whether a joint is labeled; it
does not distinguish visibility from occlusion. Missing joints (v=0) receive
confidence target 0 and no coordinate supervision.
"""

import math

import torch
import torch.nn.functional as F

from yolo.tools.loss_functions import YOLOLoss
from yolo.utils.pose_utils import decode_pose_coordinates

# pycocotools COCOeval keypoint sigmas, in the standard 17-joint order.
COCO_OKS_SIGMAS = (
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
)


def _weighted_pose_mean(per_positive, weights, eligible):
    """TAL-weighted positive reduction; unlabeled positives do not dilute xy loss."""
    weights = weights.detach().float() * eligible
    return (per_positive * weights).sum() / weights.sum().clamp_min(1.0)


class PoseOKSLoss:
    """One minus labeled-joint OKS with official COCO area/sigma normalization.

    ``areas`` are transformed annotation areas, not predicted or GT box areas.
    Optional exact resize gains undo anisotropic stretch before measuring OKS.
    """

    def __init__(self, sigmas):
        self.sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
        if (
            self.sigmas.ndim != 1
            or not self.sigmas.numel()
            or not (torch.isfinite(self.sigmas) & (self.sigmas > 0)).all()
        ):
            raise ValueError("OKS sigmas must be a nonempty sequence of finite positive numbers")

    def __call__(self, predicted_xy, target_xy, labeled, areas, weights, gains=None):
        if predicted_xy.shape != target_xy.shape or predicted_xy.shape[-2:] != (self.sigmas.numel(), 2):
            raise ValueError("OKS coordinates must match the configured joint sigmas")
        # Mask before arithmetic so v=0 NaN placeholders cannot poison gradients.
        residual = torch.where(labeled[..., None], predicted_xy.float() - target_xy.float(), 0.0)
        areas = areas.float()
        if gains is not None:
            residual = residual / gains.float()[:, None, :]
            areas = areas / gains.float().prod(-1)
        variances = (2 * self.sigmas.to(predicted_xy.device)).square()
        exponent = residual.square().sum(-1) / (2 * variances[None] * areas[:, None].clamp_min(1e-9))
        # expm1 retains precision when a prediction is already near its target.
        joint_loss = -torch.expm1(-exponent)
        counts = labeled.sum(-1)
        per_positive = (joint_loss * labeled).sum(-1) / counts.clamp_min(1)
        return _weighted_pose_mean(per_positive, weights, counts > 0)


class PoseDFLoss:
    """Two-adjacent-bin cross entropy on a closed signed offset interval.

    Coordinates outside the representable interval are excluded and counted,
    never clipped into endpoint targets. Each labeled x/y coordinate counts
    separately, so the reported coverage has an explicit denominator.
    """

    def __init__(self, bins, pose_range):
        self.bins = int(bins)
        self.pose_range = float(pose_range)
        if self.bins < 2 or not math.isfinite(self.pose_range) or self.pose_range <= 0:
            raise ValueError("Pose DFL needs at least 2 bins and a finite positive pose_range")

    def __call__(self, logits, offsets, labeled, weights=None):
        if logits.shape[:-1] != offsets.shape or logits.shape[-1] != self.bins:
            raise ValueError("Pose logits and target offset shapes do not match")
        labeled = labeled.expand_as(offsets)
        in_range = torch.isfinite(offsets) & (offsets >= -self.pose_range) & (offsets <= self.pose_range)
        supervised = labeled & in_range
        eligible_count = labeled.sum()
        excluded_count = (labeled & ~in_range).sum()
        # Index before converting targets to integers: absent/nonfinite joints
        # must never enter cross entropy, even when all joints are absent.
        values = offsets[supervised].float()
        predictions = logits[supervised].float()
        if values.numel() == 0:
            return logits.float().sum() * 0.0, excluded_count, eligible_count
        positions = (values + self.pose_range) * ((self.bins - 1) / (2 * self.pose_range))
        # Clamp only floating-point bin-rounding error after the strict range
        # check above. The +range endpoint maps exactly to the final bin.
        positions = positions.clamp(0, self.bins - 1)
        left = positions.floor().long()
        right = (left + 1).clamp_max(self.bins - 1)
        right_weight = positions - left
        loss = F.cross_entropy(predictions, left, reduction="none") * (1 - right_weight)
        loss += F.cross_entropy(predictions, right, reduction="none") * right_weight
        if weights is None:
            return loss.mean(), excluded_count, eligible_count
        # Average each positive's supervised coordinates before applying TAL,
        # rather than giving people with more labeled joints extra weight.
        coordinate_loss = torch.zeros_like(offsets, dtype=torch.float32).masked_scatter(supervised, loss)
        counts = supervised.flatten(1).sum(-1)
        per_positive = coordinate_loss.flatten(1).sum(-1) / counts.clamp_min(1)
        return _weighted_pose_mean(per_positive, weights, counts > 0), excluded_count, eligible_count


class DualPoseLoss:
    """Detection and pose losses sharing exactly one assignment per head."""

    def __init__(self, cfg, vec2box):
        loss_cfg = cfg.task.loss
        pose_cfg = vec2box.pose_config
        self.num_keypoints = int(pose_cfg["num_keypoints"])
        self.vec2box = vec2box
        self.loss = YOLOLoss(loss_cfg, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max)
        self.pose_dfl = PoseDFLoss(pose_cfg["pose_bins"], pose_cfg["pose_range"])
        oks_rate = loss_cfg.objective.get("PoseOKSLoss", 7.5)
        self.pose_oks = None
        if oks_rate:
            sigmas = getattr(loss_cfg, "oks_sigmas", None)
            if sigmas is None:
                if self.num_keypoints != 17:
                    raise ValueError("Non-COCO pose layouts require task.loss.oks_sigmas for every joint")
                sigmas = COCO_OKS_SIGMAS
            self.pose_oks = PoseOKSLoss(sigmas)
            if self.pose_oks.sigmas.numel() != self.num_keypoints:
                raise ValueError("task.loss.oks_sigmas must contain one sigma per keypoint")
        self.aux_rate = loss_cfg.aux
        self.rates = (
            loss_cfg.objective["BoxLoss"],
            loss_cfg.objective["DFLoss"],
            loss_cfg.objective["BCELoss"],
            oks_rate,
            loss_cfg.objective.get("PoseDFLoss", 1.0),
            loss_cfg.objective.get("PoseVisibilityLoss", 1.0),
        )

    def _head_loss(self, predicts, targets):
        predicts_cls, predicts_anc, predicts_box, pose_logits, visibility_logits, _ = predicts
        # Padding is identified by class=-1, independently of the contents of
        # its remaining columns. Zero boxes cannot become TAL positives.
        detection_targets = torch.where(targets[..., :1] >= 0, targets[..., :5], 0.0)
        matched, valid, target_indices = self.loss.matcher(
            detection_targets, (predicts_cls.detach(), predicts_box.detach()), return_indices=True
        )
        targets_cls, targets_box = self.loss.separate_anchor(matched)
        cls_norm = targets_cls.sum().clamp_min(1)
        box_norm = targets_cls.sum(-1)[valid]
        scaled_boxes = predicts_box / self.vec2box.scaler[None, :, None]
        # Dynamic image sizes can update the converter's anchor grid.
        self.loss.dfl.anchors_norm = (self.vec2box.anchor_grid / self.vec2box.scaler[:, None])[None]
        detection = (
            self.loss.iou(scaled_boxes, targets_box, valid, box_norm, cls_norm),
            self.loss.dfl(predicts_anc, targets_box, valid, box_norm, cls_norm),
            self.loss.cls(predicts_cls, targets_cls, cls_norm),
        )
        batch_indices, anchor_indices = valid.nonzero(as_tuple=True)
        selected_targets = targets[batch_indices, target_indices[valid]]
        joint_end = 5 + 3 * self.num_keypoints
        joint_targets = selected_targets[:, 5:joint_end].reshape(-1, self.num_keypoints, 3)
        labeled = joint_targets[..., 2] > 0
        if targets.shape[-1] == joint_end + 3:
            areas, gains = selected_targets[:, joint_end], selected_targets[:, joint_end + 1 :]
            if not (torch.isfinite(selected_targets[:, joint_end:]) & (selected_targets[:, joint_end:] > 0)).all():
                raise ValueError("Pose target area and resize gains must be finite and positive")
        else:
            # Compatibility for direct callers with the legacy target layout.
            # The COCO loader always supplies authoritative annotation areas.
            areas = (selected_targets[:, 3:5] - selected_targets[:, 1:3]).prod(-1)
            gains = None
        offsets = (
            joint_targets[..., :2].float() - self.vec2box.anchor_grid[anchor_indices, None].float()
        ) / self.vec2box.scaler[anchor_indices, None, None]
        selected_logits = pose_logits[valid].reshape(-1, self.num_keypoints, 2, self.pose_dfl.bins)
        pose_dfl, excluded, eligible = self.pose_dfl(selected_logits, offsets, labeled[..., None], box_norm)
        pose_oks = selected_logits.float().sum() * 0.0
        if self.pose_oks is not None:
            predicted_xy = decode_pose_coordinates(
                selected_logits,
                self.vec2box.anchor_grid[anchor_indices],
                self.vec2box.scaler[anchor_indices],
                self.pose_dfl.pose_range,
            )
            pose_oks = self.pose_oks(predicted_xy, joint_targets[..., :2], labeled, areas, box_norm, gains)
        selected_visibility = visibility_logits[valid].float()
        if selected_visibility.numel():
            per_positive = F.binary_cross_entropy_with_logits(
                selected_visibility, labeled.float(), reduction="none"
            ).mean(-1)
            visibility = _weighted_pose_mean(per_positive, box_norm, torch.ones_like(box_norm, dtype=torch.bool))
        else:
            visibility = visibility_logits.float().sum() * 0.0
        return (*detection, pose_oks, pose_dfl, visibility), excluded, eligible

    def __call__(self, aux_predicts, main_predicts, targets):
        joint_end = 5 + 3 * self.num_keypoints
        if targets.shape[-1] not in (joint_end, joint_end + 3):
            raise ValueError("Pose targets require class, xyxy, joint triplets and optional area/gain_x/gain_y")
        main, excluded, eligible = self._head_loss(main_predicts, targets)
        if aux_predicts is not None:
            aux, aux_excluded, aux_eligible = self._head_loss(aux_predicts, targets)
            excluded = excluded + aux_excluded
            eligible = eligible + aux_eligible
        else:
            aux = [value * 0.0 for value in main]
        combined = [rate * (m + self.aux_rate * a) for rate, m, a in zip(self.rates, main, aux)]
        names = ("BoxLoss", "DFLLoss", "BCELoss", "PoseOKSLoss", "PoseDFLoss", "PoseVisibilityLoss")
        metrics = {f"Loss/{name}": value.detach() for name, value in zip(names, combined)}
        metrics["Pose/OutOfRangeCoordinates"] = excluded.detach().float()
        metrics["Pose/LabeledCoordinates"] = eligible.detach().float()
        metrics["Pose/OutOfRangeFraction"] = excluded.detach().float() / eligible.clamp_min(1)
        return sum(combined), metrics
