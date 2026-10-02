"""Cross-feature safeguards after integrating detection features into pose."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.conftest import get_cfg
from yolo.model.yolo import create_model
from yolo.tools.loss_functions import create_loss_function


def test_pose_rejects_nms_free_before_building_detection_only_heads():
    cfg = get_cfg(["model=v9-t-pose", "weight=false"])
    cfg.model.nms_free = True
    with pytest.raises(ValueError, match="Pose models do not support model.nms_free"):
        create_model(deepcopy(cfg.model), weight_path=False, class_num=1)


def test_pose_converter_cannot_route_into_detection_only_nms_free_loss():
    cfg = get_cfg(["model=v9-t-pose", "weight=false"])
    cfg.model.nms_free = True
    converter = SimpleNamespace(pose_config=dict(cfg.model.pose))
    with pytest.raises(ValueError, match="Pose models do not support model.nms_free"):
        create_loss_function(cfg, converter)


def test_pose_rejects_qat_training_before_weight_download():
    cfg = get_cfg(["model=v9-t-pose", "weight=false"])
    with pytest.raises(ValueError, match="do not support QAT training"):
        create_model(deepcopy(cfg.model), weight_path=False, class_num=1, qat_cfg=SimpleNamespace(enabled=True))


def test_pose_rejects_qat_checkpoint_before_preparing_quantizers():
    cfg = get_cfg(["model=v9-t-pose", "weight=false"])
    model = create_model(deepcopy(cfg.model), weight_path=False, class_num=1)
    with pytest.raises(ValueError, match="do not support QAT checkpoints"):
        model.save_load_weights({"qat": {}, "model_state_dict": model.state_dict(), "pose_config": model.pose_config})
    assert not hasattr(model, "qat_metadata")
