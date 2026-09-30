import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss

from yolo.config.config import Config, LossConfig
from yolo.utils.bounding_box_utils import BoxMatcher, Vec2Box, calculate_iou
from yolo.utils.logger import logger


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Refactor the device, should be assign by config
        # TODO: origin v9 assing pos_weight == 1?
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Any:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicts_bbox: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou", aligned=True)
        loss_iou = 1.0 - iou
        loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou


class DFLoss(nn.Module):
    def __init__(self, vec2box: Vec2Box, reg_max: int) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self, predicts_anc: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1).clamp(
            0, self.reg_max - 1.01
        )
        picked_targets = targets_dist[valid_bbox].view(-1)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = F.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_dfl = (loss_dfl * box_norm).sum() / cls_norm
        return loss_dfl


class OneToOneMatcher(BoxMatcher):
    """Task-aligned top-1 assignment for an independent NMS-free head.

    Each GT nominates a single anchor using the same class/IoU exponents as
    the one-to-many matcher. Anchor collisions retain only the nominated GT
    with the highest IoU; losing GTs can remain unmatched. All ties resolve
    to the first index. As in the original matcher, a GT with no usable grid
    candidate may fall back to its best positive alignment score.
    """

    def __init__(self, cfg, class_num: int, vec2box, reg_max: int) -> None:
        super().__init__(cfg, class_num, vec2box, reg_max)
        self.topk = 1

    @torch.no_grad()
    def __call__(self, target: Tensor, predict: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # Float16 alignment powers (IoU**6 by default) easily underflow.
        predict_cls, predict_bbox = (value.detach().float() for value in predict)
        target = target.detach().float()
        batch, anchors, _ = predict_cls.shape
        aligned = predict_cls.new_zeros((batch, anchors, self.class_num + 4))
        valid_mask = torch.zeros((batch, anchors), dtype=torch.bool, device=predict_cls.device)
        if target.shape[1] == 0 or anchors == 0:
            return aligned, valid_mask

        target_cls = target[..., 0]
        target_bbox = target[..., 1:]
        valid_targets = (
            torch.isfinite(target).all(-1)
            & (target_cls >= 0)
            & (target_cls < self.class_num)
            & (target_cls == target_cls.floor())
            & (target_bbox[..., 2:] > target_bbox[..., :2]).all(-1)
        )
        # Exclude padded/invalid rows from both the grid and fallback paths.
        target_bbox = torch.where(valid_targets[..., None], target_bbox, 0)
        target_cls = torch.where(valid_targets, target_cls, 0).long()[..., None]
        grid_mask = self.get_valid_matrix(target_bbox) & valid_targets[..., None]
        iou_mat = self.get_iou_matrix(predict_bbox, target_bbox)
        cls_mat = self.get_cls_matrix(predict_cls.sigmoid(), target_cls)
        metric = iou_mat.pow(self.factor["iou"]) * cls_mat.pow(self.factor["cls"])
        metric = metric.masked_fill(~valid_targets[..., None], 0)

        values, indices = metric.masked_fill(~grid_mask, 0).max(dim=-1)
        fallback_values, fallback_indices = metric.max(dim=-1)
        indices = torch.where(values > 0, indices, fallback_indices)
        has_match = valid_targets & (torch.maximum(values, fallback_values) > 0)
        nominations = torch.zeros_like(metric, dtype=torch.bool)
        nominations.scatter_(-1, indices[..., None], has_match[..., None])

        # Restrict the winner to nominated GTs, including in zero-IoU ties.
        winner = iou_mat.masked_fill(~nominations, -1).argmax(dim=1)
        valid_mask = nominations.any(dim=1)
        boxes = target_bbox.gather(1, winner[..., None].expand(-1, -1, 4))
        classes = target_cls.gather(1, winner[..., None])
        quality = iou_mat.gather(1, winner[:, None, :]).squeeze(1)
        # For top-1, normalized task alignment reduces to the selected IoU.
        # Avoid dividing very small alignment scores by a fixed epsilon.
        scores = predict_cls.new_zeros(predict_cls.shape)
        scores.scatter_(-1, classes, (quality * valid_mask)[..., None])
        boxes = boxes.masked_fill(~valid_mask[..., None], 0)
        return torch.cat((scores, boxes), dim=-1), valid_mask


class YOLOLoss:
    def __init__(self, loss_cfg: LossConfig, vec2box: Vec2Box, class_num: int = 80, reg_max: int = 16) -> None:
        self.class_num = class_num
        self.vec2box = vec2box

        self.cls = BCELoss()
        self.dfl = DFLoss(vec2box, reg_max)
        self.iou = BoxLoss()

        self.matcher = BoxMatcher(loss_cfg.matcher, self.class_num, vec2box, reg_max)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicts_cls, predicts_anc, predicts_box = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets, (predicts_cls.detach(), predicts_box.detach()))

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        cls_norm = targets_cls.sum().clamp_min(1)
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.iou(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        return loss_iou, loss_dfl, loss_cls


class DualLoss:
    def __init__(self, cfg: Config, vec2box) -> None:
        loss_cfg = cfg.task.loss
        self.loss = YOLOLoss(loss_cfg, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max)

        self.aux_rate = loss_cfg.aux

        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]

    def __call__(
        self, aux_predicts: List[Tensor], main_predicts: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # TODO: Need Refactor this region, make it flexible!
        if aux_predicts is None:
            aux_iou, aux_dfl, aux_cls = 0, 0, 0
        else:
            aux_iou, aux_dfl, aux_cls = self.loss(aux_predicts, targets)
        main_iou, main_dfl, main_cls = self.loss(main_predicts, targets)

        total_loss = [
            self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            self.cls_rate * (aux_cls * self.aux_rate + main_cls),
        ]
        loss_dict = {f"Loss/{name}Loss": value.detach() for name, value in zip(["Box", "DFL", "BCE"], total_loss)}
        return sum(total_loss), loss_dict


class NMSFreeLoss(DualLoss):
    """Keep dense/auxiliary supervision and train the independent top-1 head."""

    def __init__(self, cfg: Config, vec2box) -> None:
        super().__init__(cfg, vec2box)
        self.one2one_rate = float(cfg.task.loss.one2one)
        if not math.isfinite(self.one2one_rate) or self.one2one_rate <= 0:
            raise ValueError("NMS-free loss one2one weight must be finite and positive")
        self.one2one_loss = YOLOLoss(
            cfg.task.loss, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max
        )
        self.one2one_loss.matcher = OneToOneMatcher(
            cfg.task.loss.matcher, cfg.dataset.class_num, vec2box, cfg.model.anchor.reg_max
        )

    def __call__(
        self,
        aux_predicts: Optional[List[Tensor]],
        main_predicts: List[Tensor],
        targets: Tensor,
        one2one_predicts: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if one2one_predicts is None:
            raise ValueError("NMS-free training requires one2one_predicts from the independent head")
        dense_loss, loss_dict = super().__call__(aux_predicts, main_predicts, targets)
        one2one_terms = self.one2one_loss(one2one_predicts, targets)
        weighted = [
            self.one2one_rate * rate * term
            for rate, term in zip((self.iou_rate, self.dfl_rate, self.cls_rate), one2one_terms)
        ]
        one2one_loss = sum(weighted)
        for name, value in zip(("Box", "DFL", "BCE"), weighted):
            loss_dict[f"Loss/{name}Loss"] = loss_dict[f"Loss/{name}Loss"] + value.detach()
        loss_dict["Loss/One2ManyLoss"] = dense_loss.detach()
        loss_dict["Loss/One2OneLoss"] = one2one_loss.detach()
        return dense_loss + one2one_loss, loss_dict


def create_loss_function(cfg: Config, vec2box) -> DualLoss:
    # TODO: make it flexible, if cfg doesn't contain aux, only use SingleLoss
    loss_function = NMSFreeLoss(cfg, vec2box) if getattr(cfg.model, "nms_free", False) else DualLoss(cfg, vec2box)
    logger.info(":white_check_mark: Success load loss function")
    return loss_function
