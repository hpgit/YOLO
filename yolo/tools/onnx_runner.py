"""Hydra adapter for the standalone ONNX inference file."""

from pathlib import Path

from yolo.tools.onnx_inference import ONNXDetector, run_inference
from yolo.utils.logger import logger
from yolo.utils.logging_utils import validate_log_directory


def run_onnx_inference(cfg):
    if not isinstance(cfg.weight, str) or Path(cfg.weight).suffix.lower() != ".onnx":
        raise ValueError("ONNX inference requires weight=/path/to/exported.onnx from task=export.")
    options = cfg.task.onnx
    detector = ONNXDetector(
        cfg.weight,
        providers=list(options.providers),
        output_format=options.output_format,
        class_num=cfg.dataset.class_num,
        reg_max=getattr(cfg.model.anchor, "reg_max", 16),
        strides=list(options.strides),
        confidence=cfg.task.nms.min_confidence,
        iou_threshold=cfg.task.nms.min_iou,
        max_detections=cfg.task.nms.max_bbox,
        threads=cfg.cpu_num,
        nms_free=getattr(cfg.model, "nms_free", False),
    )
    # New metadata is authoritative. Old exports use the matching dataset config.
    if "yolo.inference" not in detector.session.get_modelmeta().custom_metadata_map:
        if len(cfg.dataset.class_list) == detector.class_num:
            detector.class_names = list(cfg.dataset.class_list)
    output = validate_log_directory(cfg, cfg.name) if cfg.task.save_predict else None
    count = run_inference(detector, cfg.task.data.source, output)
    logger.info(f"ONNX inference processed {count} frames" + (f"; saved results to {output}" if output else ""))
    return count
