"""Compatibility import for the measured NMS candidate, now used in production."""

from yolo.utils.nms_utils import grouped_batched_nms

__all__ = ["grouped_batched_nms"]
