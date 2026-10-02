"""Detection selection for NMS-based and one-to-one trained detectors."""

import torch
from torchvision.ops import batched_nms, nms


def grouped_batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, groups: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """Preserve torchvision NMS ordering while grouping large inputs once."""
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return batched_nms(boxes, scores, groups, iou_threshold)
    threshold = 4000 if boxes.device.type == "cpu" else 20000
    if boxes.numel() <= threshold:
        return batched_nms(boxes, scores, groups, iou_threshold)
    # Preserve original input order within each class, including tied scores.
    order = torch.argsort(groups, stable=True)
    _, counts = torch.unique_consecutive(groups[order], return_counts=True)
    sorted_boxes, sorted_scores = boxes[order], scores[order]
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    start = 0
    for count in counts.tolist():
        end = start + count
        kept = nms(sorted_boxes[start:end], sorted_scores[start:end], iou_threshold)
        keep_mask[order[start:end][kept]] = True
        start = end
    indices = torch.where(keep_mask)[0]
    return indices[scores[indices].sort(descending=True)[1]]


def select_nms_free_detections(
    boxes: torch.Tensor, scores: torch.Tensor, confidence: float = 0.5, max_detections: int = 300
) -> list[torch.Tensor]:
    """Choose each anchor's best class, then stable top-k independently per image.

    Scores must already be probabilities. No overlap-based suppression or
    multi-label expansion is applied; use only with a one-to-one trained head.
    Returns rows [class_id, x1, y1, x2, y2, score], including empty (0, 6) tensors.
    """
    if not 0 <= confidence <= 1 or max_detections < 1:
        raise ValueError("Confidence must be in [0,1] and max_detections must be positive.")
    best_scores, class_ids = scores.max(dim=-1)
    valid = (
        torch.isfinite(boxes).all(dim=-1)
        & (boxes[..., 2:] > boxes[..., :2]).all(dim=-1)
        & torch.isfinite(best_scores)
        & (best_scores > confidence)
    )
    results = []
    for image_boxes, image_scores, image_classes, image_valid in zip(boxes, best_scores, class_ids, valid):
        indices = torch.where(image_valid)[0]
        order = torch.argsort(image_scores[indices], descending=True, stable=True)[:max_detections]
        indices = indices[order]
        results.append(
            torch.cat((image_classes[indices, None], image_boxes[indices], image_scores[indices, None]), dim=-1)
        )
    return results
