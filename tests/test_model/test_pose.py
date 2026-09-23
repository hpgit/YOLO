"""Pose heads preserve detection weights and expose raw coordinate distributions."""

from copy import deepcopy

import pytest
import torch

from tests.conftest import get_cfg
from yolo.model.module import MultiheadDetection, MultiheadPose, PoseDetection
from yolo.model.yolo import create_model


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("variant", ["t", "s", "m", "c"])
def test_pose_configs_compose_and_forward_both_branches(variant):
    cfg = get_cfg([f"model=v9-{variant}-pose", "weight=false"])
    model = create_model(deepcopy(cfg.model), class_num=1, weight_path=False).eval()
    assert model.pose_config == {"num_keypoints": 17, "pose_bins": 64, "pose_range": 32.0}
    assert isinstance(model.model[model.layer_index["Main"] - 1], MultiheadPose)
    assert isinstance(model.model[model.layer_index["AUX"] - 1], MultiheadPose)
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 64, 64))
    assert set(outputs) == {"Main", "AUX"}
    for branch in outputs.values():
        for size, (classes, anchors, boxes, poses, visibility) in zip((8, 4, 2), branch):
            assert classes.shape == (1, 1, size, size)
            assert anchors.shape == (1, 16, 4, size, size)
            assert boxes.shape == (1, 4, size, size)
            assert poses.shape == (1, 17 * 2 * 64, size, size)
            assert visibility.shape == (1, 17, size, size)


def test_pose_configuration_reaches_main_and_auxiliary():
    cfg = get_cfg(
        ["model=v9-t-pose", "model.pose.num_keypoints=5", "model.pose.pose_bins=12", "model.pose.pose_range=7.5"]
    )
    model = create_model(deepcopy(cfg.model), class_num=2, weight_path=False)
    heads = [layer for layer in model.modules() if isinstance(layer, PoseDetection)]
    assert len(heads) == 6
    assert all((head.num_keypoints, head.pose_bins, head.pose_range) == (5, 12, 7.5) for head in heads)


def test_pose_coordinates_backpropagate_without_box_branch_dependency():
    head = PoseDetection((16, 24), 2, num_keypoints=3, pose_bins=8, pose_range=4)
    features = torch.randn(2, 24, 4, 5, requires_grad=True)
    classes, anchors, boxes, logits, visibility = head(features)
    assert logits.shape == (2, 48, 4, 5)
    assert visibility.shape == (2, 3, 4, 5)
    (logits.square().mean() + visibility.square().mean()).backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    assert head.pose_logits.weight.grad.abs().sum() > 0
    assert head.visibility_conv.weight.grad.abs().sum() > 0
    assert head.anchor_conv[-1].weight.grad is None
    assert head.class_conv[-1].weight.grad is None


def test_detection_parameters_transfer_without_rewriting_paths():
    detection = MultiheadDetection([16, 24], 2).eval()
    pose = MultiheadPose([16, 24], 2, num_keypoints=3, pose_bins=8).eval()
    result = pose.load_state_dict(detection.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys
    assert all(
        any(name in key for name in ("pose_conv", "pose_logits", "visibility_conv")) for key in result.missing_keys
    )
    features = [torch.randn(2, 16, 4, 4), torch.randn(2, 24, 2, 2)]
    with torch.no_grad():
        original, extended = detection(features), pose(features)
    for baseline, new in zip(original, extended):
        assert len(baseline) == 3 and len(new) == 5
        for expected, actual in zip(baseline, new[:3]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_detection_model_has_no_pose_metadata():
    cfg = get_cfg(["model=v9-t"])
    assert create_model(deepcopy(cfg.model), class_num=1, weight_path=False).pose_config is None


def test_pose_has_no_implicit_pretrained_download():
    cfg = get_cfg(["model=v9-t-pose"])
    with pytest.raises(ValueError, match="explicit checkpoint path or weight=false"):
        create_model(deepcopy(cfg.model), class_num=1, weight_path=True)


def test_partial_detection_transfer_reports_missing_pose_parameters():
    detection_cfg = get_cfg(["model=v9-t"])
    pose_cfg = get_cfg(["model=v9-t-pose"])
    detection = create_model(deepcopy(detection_cfg.model), class_num=1, weight_path=False)
    pose = create_model(deepcopy(pose_cfg.model), class_num=1, weight_path=False)
    pose.save_load_weights(detection.model.state_dict())
    assert not pose.weight_load_report["mismatch"]
    assert pose.weight_load_report["missing"]
    assert all(
        any(part in key for part in ("pose_conv", "pose_logits", "visibility_conv"))
        for key in pose.weight_load_report["missing"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_keypoints": 0},
        {"num_keypoints": 1.5},
        {"pose_bins": 1},
        {"pose_bins": 2.5},
        {"pose_range": 0},
        {"pose_range": float("inf")},
        {"pose_range": float("nan")},
    ],
)
def test_invalid_pose_dimensions_fail_early(kwargs):
    with pytest.raises(ValueError):
        PoseDetection((16, 24), 1, **kwargs)
