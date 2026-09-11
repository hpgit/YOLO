"""NMS grouping without repeatedly scanning the full candidate tensor."""
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
