"""Pose contracts survive detection QAT and export metadata integration."""

import json
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from tests.test_tools.test_pose_export import pose_config
from yolo.tools.export import export_model
from yolo.utils.checkpoint_utils import YOLOCheckpoint


def test_best_pose_weights_preserve_layout_and_checkpoint_state(tmp_path, monkeypatch):
    detector = torch.nn.Module()
    detector.model = torch.nn.Linear(2, 1)
    detector.pose_config = {"num_keypoints": 3, "pose_bins": 8, "pose_range": 4}
    monkeypatch.setattr(ModelCheckpoint, "on_validation_end", lambda *_: None)
    trainer = SimpleNamespace(
        state=SimpleNamespace(fn="fit"),
        sanity_checking=False,
        callback_metrics={"map": torch.tensor(0.5)},
        is_global_zero=True,
    )
    callback = YOLOCheckpoint(tmp_path)
    callback.on_validation_end(trainer, SimpleNamespace(ema=detector))
    payload = torch.load(tmp_path / "best.pt", weights_only=True)
    assert payload["pose_config"] == detector.pose_config
    key = "weights"
    expected = detector.model.state_dict()
    assert payload[key].keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(payload[key][name], tensor)
    assert "qat" not in payload


def test_pose_export_metadata_keeps_box_bins_and_distinguishes_pose(tmp_path):
    onnx = pytest.importorskip("onnx")
    cfg = pose_config(f"task.output={tmp_path / 'pose.onnx'}")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        path = export_model(cfg)
    finally:
        torch.set_num_threads(previous)
    props = {prop.key: prop.value for prop in onnx.load(path).metadata_props}
    metadata = json.loads(props["yolo.inference"])
    assert metadata["reg_max"] == cfg.model.anchor.reg_max
    assert metadata["pose_config"] == dict(cfg.model.pose)
    assert metadata["output_format"] == "pose_dfl"
    assert metadata["nms_free"] is False

    pytest.importorskip("onnxruntime")
    from yolo.tools.onnx_inference import ONNXDetector

    with pytest.raises(ValueError, match="Portable ONNX inference does not support pose models"):
        ONNXDetector(path, threads=1)
