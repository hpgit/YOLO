"""Grid-relative pose decoding and instance-preserving postprocessing."""

import torch

from yolo.utils.bounding_box_utils import Vec2Box
from yolo.utils.nms_utils import grouped_batched_nms


class Pose2Box(Vec2Box):
    """Decode signed keypoint offsets independently of predicted boxes.

    Returns cls, box distributions, boxes, pose distributions [B,N,2K,M],
    keypoint confidence logits [B,N,K], and keypoints [B,N,K,3].
    """

    def __init__(self, model, anchor_cfg, image_size, device):
        super().__init__(model, anchor_cfg, image_size, device)
        self.pose_config = model.pose_config
        self.num_keypoints = self.pose_config["num_keypoints"]
        self.pose_bins = self.pose_config["pose_bins"]
        self.pose_range = self.pose_config["pose_range"]

    def __call__(self, predicts):
        classes, anchors, boxes = super().__call__([head[:3] for head in predicts])
        logits = torch.cat([head[3].flatten(2).transpose(1, 2) for head in predicts], 1)
        logits = logits.reshape(logits.shape[0], -1, 2 * self.num_keypoints, self.pose_bins)
        visibility = torch.cat([head[4].flatten(2).transpose(1, 2) for head in predicts], 1)
        # Accumulate expectations in float32 even under mixed precision.
        bins = torch.linspace(-self.pose_range, self.pose_range, self.pose_bins, device=logits.device)
        offsets = (logits.float().softmax(-1) * bins).sum(-1)
        offsets = offsets.reshape(offsets.shape[0], -1, self.num_keypoints, 2)
        xy = self.anchor_grid[None, :, None, :] + offsets * self.scaler[None, :, None, None]
        keypoints = torch.cat((xy, visibility.float().sigmoid()[..., None]), -1)
        return classes, anchors, boxes, logits, visibility, keypoints


def pose_nms(classes, boxes, keypoints, nms_cfg):
    """Keep each full skeleton attached to its exact selected detection grid.

    Result rows: class, xyxy, object score, followed by K interleaved x,y,score.
    """
    scores = classes.float().sigmoid()
    batch, grid, label = torch.where(scores > nms_cfg.min_confidence)
    candidate_scores = scores[batch, grid, label]
    candidate_boxes = boxes[batch, grid].float()
    keep = grouped_batched_nms(candidate_boxes, candidate_scores, batch + label * boxes.shape[0], nms_cfg.min_iou)
    results = []
    for index in range(boxes.shape[0]):
        selected = keep[batch[keep] == index][: nms_cfg.max_bbox]
        results.append(
            torch.cat(
                (
                    label[selected, None].to(candidate_boxes.dtype),
                    candidate_boxes[selected],
                    candidate_scores[selected, None],
                    keypoints[batch[selected], grid[selected]].flatten(1),
                ),
                -1,
            )
        )
    return results


def reverse_pose_coordinates(boxes, keypoints, reverse):
    """Undo letterbox without modifying confidence or the source tensors.

    Accept legacy [gain,pad_x,pad_y,pad_x,pad_y] or exact pose transform
    [gain_x,gain_y,pad_x,pad_y].
    """
    reverse = reverse.to(device=boxes.device, dtype=boxes.dtype)
    if reverse.shape[-1] == 4:
        gains, pads = reverse[:, :2], reverse[:, 2:]
    elif reverse.shape[-1] == 5:
        gains, pads = reverse[:, :1].expand(-1, 2), reverse[:, 1:3]
    else:
        raise ValueError("Pose inverse transform requires four or five values per image.")
    boxes = (boxes - pads.repeat(1, 2)[:, None]) / gains.repeat(1, 2)[:, None]
    xy = (keypoints[..., :2] - pads[:, None, None]) / gains[:, None, None]
    return boxes, torch.cat((xy, keypoints[..., 2:]), -1)
