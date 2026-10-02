from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from yolo.tools.loss_functions import create_loss_function
from yolo.tools.pose_loss import DualPoseLoss, PoseDFLoss
from yolo.utils.bounding_box_utils import BoxMatcher


def fixture_parts():
    converter = SimpleNamespace(
        anchor_grid=torch.tensor([[4.0, 4.0], [12.0, 4.0], [20.0, 4.0], [28.0, 4.0]]),
        scaler=torch.full((4,), 8.0),
        pose_config={"num_keypoints": 2, "pose_bins": 5, "pose_range": 2.0},
    )
    matcher = {"iou": "ciou", "topk": 2, "factor": {"iou": 6.0, "cls": 0.5}}
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            loss=SimpleNamespace(
                matcher=matcher,
                aux=0.25,
                objective={"BoxLoss": 7.5, "DFLoss": 1.5, "BCELoss": 0.5, "PoseDFLoss": 2.0, "PoseVisibilityLoss": 0.5},
            )
        ),
        model=SimpleNamespace(anchor=SimpleNamespace(reg_max=16)),
        dataset=SimpleNamespace(class_num=1),
    )
    targets = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 16.0, 8.0, 4.0, 4.0, 2.0, 10.0, 3.0, 1.0],
                [0.0, 16.0, 0.0, 32.0, 8.0, 20.0, 4.0, 2.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    boxes = torch.tensor(
        [[[0.0, 0.0, 16.0, 8.0], [0.0, 0.0, 16.0, 8.0], [16.0, 0.0, 32.0, 8.0], [16.0, 0.0, 32.0, 8.0]]],
        requires_grad=True,
    )
    predictions = [
        torch.zeros(1, 4, 1, requires_grad=True),
        torch.zeros(1, 4, 4, 16, requires_grad=True),
        boxes,
        torch.zeros(1, 4, 4, 5, requires_grad=True),
        torch.zeros(1, 4, 2, requires_grad=True),
        torch.zeros(1, 4, 2, 3),
    ]
    return cfg, converter, targets, predictions


def test_pose_dfl_interpolation_endpoints_and_gradients():
    criterion = PoseDFLoss(5, 2.0)
    logits = torch.tensor([[2.0, 0.0, 1.0, -1.0, 3.0]] * 3, requires_grad=True)
    offsets = torch.tensor([-2.0, -0.5, 2.0])
    actual, excluded, eligible = criterion(logits, offsets, torch.ones(3, dtype=torch.bool))
    expected = (
        F.cross_entropy(logits[:1], torch.tensor([0]))
        + 0.5 * F.cross_entropy(logits[1:2], torch.tensor([1]))
        + 0.5 * F.cross_entropy(logits[1:2], torch.tensor([2]))
        + F.cross_entropy(logits[2:], torch.tensor([4]))
    ) / 3
    torch.testing.assert_close(actual, expected)
    assert excluded.item() == 0 and eligible.item() == 3
    actual.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_pose_dfl_masks_missing_and_reports_unrepresentable_coordinates():
    logits = torch.randn(5, 5, requires_grad=True)
    offsets = torch.tensor([-3.0, 3.0, float("nan"), 0.0, float("nan")])
    labeled = torch.tensor([True, True, True, True, False])
    loss, excluded, eligible = PoseDFLoss(5, 2.0)(logits, offsets, labeled)
    torch.testing.assert_close(loss, F.cross_entropy(logits[3:4], torch.tensor([2])))
    assert excluded.item() == 3 and eligible.item() == 4
    loss.backward()
    assert logits.grad[:3].count_nonzero() == 0
    assert logits.grad[4].count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pose_dfl_empty_is_finite_differentiable(dtype):
    logits = torch.randn(2, 5, dtype=dtype, requires_grad=True)
    loss, excluded, eligible = PoseDFLoss(5, 2.0)(logits, torch.zeros(2), torch.zeros(2, dtype=torch.bool))
    assert loss.dtype == torch.float32 and loss.item() == 0
    loss.backward()
    assert logits.grad.count_nonzero() == 0
    assert excluded.item() == eligible.item() == 0


def test_matcher_returns_original_target_indices_and_preserves_default_api():
    cfg, converter, targets, predictions = fixture_parts()
    matcher = BoxMatcher(cfg.task.loss.matcher, 1, converter, 16)
    legacy = matcher(targets[..., :5], (predictions[0].detach(), predictions[2].detach()))
    matched, valid, indices = matcher(targets[..., :5], (predictions[0].detach(), predictions[2].detach()), True)
    assert len(legacy) == 2 and valid.all()
    torch.testing.assert_close(matched, legacy[0])
    torch.testing.assert_close(indices, torch.tensor([[0, 0, 1, 1]]))


def test_all_tal_positives_receive_pose_gradients_and_aux_weight_is_respected():
    cfg, converter, targets, predictions = fixture_parts()
    criterion = create_loss_function(cfg, converter)
    assert isinstance(criterion, DualPoseLoss)
    main_loss, main_metrics = criterion(None, predictions, targets)
    loss, metrics = criterion(predictions, predictions, targets)
    torch.testing.assert_close(loss, main_loss * 1.25)
    for key in main_metrics:
        if key.startswith("Loss/"):
            torch.testing.assert_close(metrics[key], main_metrics[key] * 1.25)
    assert metrics["Pose/OutOfRangeFraction"].item() == 0
    loss.backward()
    assert all(torch.isfinite(tensor.grad).all() for tensor in predictions[:5])
    # Both TAL positives for each person learn pose, including the occluded
    # but labeled second joint of person 0.
    gradient = predictions[3].grad.reshape(1, 4, 2, 2, 5)
    assert (gradient[0, :, 0].abs().sum((1, 2)) > 0).all()
    assert (gradient[0, :2, 1].abs().sum((1, 2)) > 0).all()
    assert gradient[0, 2:, 1].count_nonzero() == 0
    assert (predictions[4].grad[0, :2, :] < 0).all()
    assert (predictions[4].grad[0, 2:, 1] > 0).all()


@pytest.mark.parametrize("empty_kind", ["zero_rows", "padded", "unlabeled"])
def test_pose_training_handles_empty_images_and_missing_joints(empty_kind):
    cfg, converter, targets, predictions = fixture_parts()
    if empty_kind == "zero_rows":
        targets = targets[:, :0]
    elif empty_kind == "padded":
        # Padding must remain invalid even with stale box/joint coordinates.
        targets = targets.clone()
        targets[..., 0] = -1
    else:
        targets[..., 5:] = 0
    loss, metrics = DualPoseLoss(cfg, converter)(predictions, predictions, targets)
    assert torch.isfinite(loss)
    assert metrics["Loss/PoseDFLoss"].item() == 0
    assert metrics["Pose/LabeledCoordinates"].item() == 0
    loss.backward()
    assert predictions[3].grad.count_nonzero() == 0
    if empty_kind != "unlabeled":
        assert metrics["Loss/PoseVisibilityLoss"].item() == 0
        assert predictions[4].grad.count_nonzero() == 0


def test_training_reports_out_of_range_without_silent_clipping():
    cfg, converter, targets, predictions = fixture_parts()
    targets[0, 0, 5] = 1000.0
    loss, metrics = DualPoseLoss(cfg, converter)(None, predictions, targets)
    assert torch.isfinite(loss)
    assert metrics["Pose/OutOfRangeCoordinates"].item() == 2
    assert metrics["Pose/LabeledCoordinates"].item() == 12
    torch.testing.assert_close(metrics["Pose/OutOfRangeFraction"], torch.tensor(1 / 6))
    loss.backward()
    assert predictions[3].grad[0, :2, 0].count_nonzero() == 0
