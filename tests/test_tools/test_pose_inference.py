"""Numerical pose decoding, skeleton association and checkpoint safeguards."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tests.conftest import get_cfg
from yolo.model.yolo import create_model
from yolo.tools.drawer import draw_poses
from yolo.utils.bounding_box_utils import bbox_nms, create_converter
from yolo.utils.model_utils import PostProcess
from yolo.utils.pose_utils import Pose2Box, pose_nms, reverse_pose_coordinates


def test_grid_pose_signed_endpoints_and_bbox_independence():
    model = SimpleNamespace(pose_config=dict(num_keypoints=1, pose_bins=3, pose_range=2.0))
    cfg = SimpleNamespace(strides=[8])
    decoder = Pose2Box(model, cfg, [16, 8], "cpu")
    logits = torch.full((1, 6, 1, 2), -100.0)
    logits[:, 0] = 100  # x=-2 strides
    logits[:, 5] = 100  # y=+2 strides
    head = (
        torch.zeros(1, 1, 1, 2),
        torch.zeros(1, 4, 4, 1, 2),
        torch.ones(1, 4, 1, 2),
        logits,
        torch.zeros(1, 1, 1, 2),
    )
    decoded = decoder([head])
    torch.testing.assert_close(decoded[5][0, :, 0], torch.tensor([[-12.0, 20.0, 0.5], [-4.0, 20.0, 0.5]]))
    changed = (*head[:2], head[2] * 7, *head[3:])
    torch.testing.assert_close(decoder([changed])[5], decoded[5])
    assert not torch.equal(decoder([changed])[2], decoded[2])


def test_pose_nms_preserves_class_batch_and_grid_association():
    cfg = SimpleNamespace(min_confidence=0.5, min_iou=0.5, max_bbox=3)
    classes = torch.tensor([[[5.0, 4.0], [3.0, -9.0], [-9.0, 2.0]], [[-9.0, -9.0], [7.0, -9.0], [-9.0, 6.0]]])
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]]]).repeat(2, 1, 1)
    points = torch.arange(2 * 3 * 2 * 3).reshape(2, 3, 2, 3).float()
    actual = pose_nms(classes, boxes, points, cfg)
    expected = bbox_nms(classes, boxes, cfg)
    for batch in range(2):
        torch.testing.assert_close(actual[batch][:, :6], expected[batch])
    torch.testing.assert_close(actual[0][:, 6:], points[0, [0, 0, 2]].flatten(1))
    torch.testing.assert_close(actual[1][:, 6:], points[1, [1, 2]].flatten(1))
    empty = pose_nms(torch.full_like(classes, -100), boxes, points, cfg)
    assert all(row.shape == (0, 12) for row in empty)


def test_inverse_letterbox_preserves_confidence_and_inputs():
    boxes = torch.tensor([[[7.0, 11.0, 27.0, 21.0]]])
    points = torch.tensor([[[[17.0, 16.0, 0.75]]]])
    original = points.clone()
    b, k = reverse_pose_coordinates(boxes, points, torch.tensor([[2.0, 1.0, 7.0, 11.0]]))
    torch.testing.assert_close(b, torch.tensor([[[0.0, 0.0, 10.0, 10.0]]]))
    torch.testing.assert_close(k, torch.tensor([[[[5.0, 5.0, 0.75]]]]))
    torch.testing.assert_close(points, original)


def test_draw_pose_empty_and_visibility_nonmutation():
    image = Image.new("RGB", (80, 60))
    empty = torch.empty(0, 57)
    assert draw_poses(image, empty).tobytes() == image.tobytes()
    # Degenerate bbox outside the canvas lets the joint drawing be isolated.
    rows = torch.tensor([[0.0, -10.0, -10.0, -5.0, -5.0, 0.8, 30.0, 30.0, 0.9, 50.0, 30.0, 0.1]])
    before = rows.clone()
    actual = draw_poses(image, rows, skeleton=[(0, 1)], confidence=0.5)
    assert actual.getpixel((30, 30)) != image.getpixel((30, 30))
    assert actual.getpixel((50, 30)) == image.getpixel((50, 30))
    torch.testing.assert_close(rows, before)
    assert image.getpixel((30, 30)) == (0, 0, 0)


def test_pose_checkpoint_config_and_completeness(tmp_path):
    cfg = get_cfg(["model=v9-t-pose", "weight=false", "dataset.class_num=1"])
    model = create_model(deepcopy(cfg.model), weight_path=False, class_num=1)
    path = tmp_path / "pose.pt"
    torch.save({"weights": model.model.state_dict(), "pose_config": model.pose_config}, path)
    loaded = create_model(deepcopy(cfg.model), weight_path=path, class_num=1)
    loaded.require_pose_weights()
    wrong = deepcopy(cfg.model)
    wrong.pose.pose_range = 16
    with pytest.raises(ValueError, match="pose_config"):
        create_model(wrong, weight_path=path, class_num=1)
    torch.save(model.model.state_dict(), path)
    legacy = create_model(deepcopy(cfg.model), weight_path=path, class_num=1)
    with pytest.raises(ValueError, match="metadata"):
        legacy.require_pose_weights()
