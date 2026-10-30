"""Coordinate-loss checks against official OKS and independent geometry cases."""

import math

import pytest
import torch
from pycocotools.cocoeval import COCOeval

from yolo.tools.pose_loss import PoseOKSLoss
from yolo.utils.pose_utils import decode_pose_coordinates


def test_pose_oks_matches_official_coco_with_annotation_area_and_visibility():
    evaluator = COCOeval(iouType="keypoints")
    sigmas = evaluator.params.kpt_oks_sigmas.tolist()
    xy = torch.tensor([[30.0 + index, 40.0 + index * 2] for index in range(17)])
    visibility = torch.tensor([1, 2] * 8 + [0])
    predicted = xy + torch.tensor([[index / 2 + 1, 3.0] for index in range(17)])
    predicted[-1] = 10000  # The unannotated joint must not enter OKS.
    area = 1800.0  # Deliberately different from bbox area (4000).
    target_points = torch.cat((xy, visibility[:, None]), dim=1)
    predicted_points = torch.cat((predicted, torch.ones(17, 1)), dim=1)
    evaluator._gts[(1, 1)] = [{"keypoints": target_points.flatten().tolist(), "bbox": [20, 20, 50, 80], "area": area}]
    evaluator._dts[(1, 1)] = [{"keypoints": predicted_points.flatten().tolist(), "score": 1.0}]
    expected = 1.0 - evaluator.computeOks(1, 1)[0, 0]
    predicted = predicted.unsqueeze(0).requires_grad_()
    loss = PoseOKSLoss(sigmas)(
        predicted, xy.unsqueeze(0), (visibility > 0).unsqueeze(0), torch.tensor([area]), torch.ones(1)
    )
    assert loss.item() == pytest.approx(expected, abs=2e-7)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0, -1].count_nonzero() == 0
    assert (predicted.grad[0, :-1].abs().sum(-1) > 0).all()


def test_expected_distribution_coordinates_account_for_stride_in_oks():
    # The same bin probabilities have expected x offset 0.5 at each level.
    probabilities = torch.tensor([[[0.1, 0.3, 0.6], [0.25, 0.5, 0.25]]])
    logits = probabilities.log().unsqueeze(0).repeat(3, 1, 1, 1).requires_grad_()
    anchors = torch.tensor([[4.0, 4.0], [8.0, 8.0], [16.0, 16.0]])
    strides = torch.tensor([8.0, 16.0, 32.0])
    decoded = decode_pose_coordinates(logits, anchors, strides, 1.0)
    expected = anchors[:, None, :] + torch.tensor([[[4.0, 0.0]], [[8.0, 0.0]], [[16.0, 0.0]]])
    torch.testing.assert_close(decoded, expected)
    criterion = PoseOKSLoss([0.1])
    losses = [
        criterion(
            decoded[index : index + 1],
            anchors[index : index + 1, None],
            torch.ones(1, 1, dtype=torch.bool),
            torch.tensor([10000.0]),
            torch.ones(1),
        )
        for index in range(3)
    ]
    assert 0 < losses[0] < losses[1] < losses[2] < 1
    sum(losses).backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum((1, 2, 3)) > 0).all()


def test_pose_oks_penalizes_small_people_and_precise_joints_more():
    predicted = torch.tensor([[[5.0, 0.0]]])
    target = torch.zeros_like(predicted)
    labeled = torch.ones(1, 1, dtype=torch.bool)
    scores = torch.ones(1)
    small_person = PoseOKSLoss([0.1])(predicted, target, labeled, torch.tensor([1000.0]), scores)
    large_person = PoseOKSLoss([0.1])(predicted, target, labeled, torch.tensor([4000.0]), scores)
    precise_joint = PoseOKSLoss([0.025])(predicted, target, labeled, torch.tensor([4000.0]), scores)
    assert large_person < small_person
    assert large_person < precise_joint


def test_pose_oks_tal_scores_weight_positive_means_without_score_gradients():
    predicted = torch.tensor([[[2.0, 0.0], [4.0, 0.0]], [[5.0, 0.0], [9.0, 0.0]]], requires_grad=True)
    target = torch.zeros_like(predicted)
    labeled = torch.tensor([[True, True], [True, False]])
    areas = torch.tensor([1000.0, 1000.0])
    weights = torch.tensor([0.8, 0.4], requires_grad=True)
    criterion = PoseOKSLoss([0.1, 0.1])
    first = criterion(predicted[:1], target[:1], labeled[:1], areas[:1], torch.ones(1))
    second = criterion(predicted[1:], target[1:], labeled[1:], areas[1:], torch.ones(1))
    loss = criterion(predicted, target, labeled, areas, weights)
    torch.testing.assert_close(loss, (first * 0.8 + second * 0.4) / 1.2)
    loss.backward()
    assert weights.grad is None
    assert predicted.grad[1, 1].count_nonzero() == 0
    assert predicted.grad[0].abs().sum() > 0
    assert predicted.grad[1, 0].abs().sum() > 0


def test_pose_oks_excludes_unlabeled_people_from_weight_normalizer():
    predicted = torch.tensor([[[5.0, 0.0]], [[999.0, 999.0]]])
    target = torch.zeros_like(predicted)
    labeled = torch.tensor([[True], [False]])
    areas = torch.tensor([1000.0, 1000.0])
    criterion = PoseOKSLoss([0.1])
    unit_loss = criterion(predicted[:1], target[:1], labeled[:1], areas[:1], torch.ones(1))
    loss = criterion(predicted, target, labeled, areas, torch.tensor([0.2, 100.0]))
    # Match the detection-loss low-score floor while ignoring box-only people.
    torch.testing.assert_close(loss, unit_loss * 0.2)


def test_pose_oks_masks_missing_nan_coordinates_before_distance_and_backward():
    predicted = torch.tensor([[[4.0, 5.0], [2.0, 3.0], [10.0, 10.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [float("nan"), float("nan")]]])
    visibility = torch.tensor([[1, 2, 0]])
    criterion = PoseOKSLoss([0.1, 0.1, 0.1])
    loss = criterion(predicted, target, visibility > 0, torch.tensor([1000.0]), torch.ones(1))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0, 0].abs().sum() > 0  # Occluded v=1 still supervises xy.
    assert predicted.grad[0, 1].abs().sum() > 0
    assert predicted.grad[0, 2].count_nonzero() == 0


@pytest.mark.parametrize("gains", [[2.0, 0.5], [79 / 101, 81 / 57], [79 / 101, 44 / 57]])
def test_pose_oks_is_invariant_to_anisotropic_resize(gains):
    predicted = torch.tensor([[[24.0, 33.0], [45.0, 50.0]]])
    target = torch.tensor([[[20.0, 30.0], [42.0, 48.0]]])
    area = torch.tensor([1500.0])
    gain = torch.tensor([gains])
    padding = torch.tensor([[[5.0, 18.0]]])
    labeled = torch.ones(1, 2, dtype=torch.bool)
    criterion = PoseOKSLoss([0.026, 0.079])
    original = criterion(predicted, target, labeled, area, torch.ones(1))
    resized = criterion(
        predicted * gain[:, None] + padding,
        target * gain[:, None] + padding,
        labeled,
        area * gain.prod(-1),
        torch.ones(1),
        gain,
    )
    torch.testing.assert_close(resized, original, atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("kind", ["no_positives", "unlabeled", "zero_scores", "perfect"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pose_oks_zero_loss_remains_finite_and_differentiable(kind, dtype):
    count = 0 if kind == "no_positives" else 2
    predicted = torch.full((count, 2, 2), 4.0, dtype=dtype, requires_grad=True)
    target = torch.zeros_like(predicted)
    labeled = torch.ones(count, 2, dtype=torch.bool)
    weights = torch.ones(count)
    if kind == "unlabeled":
        labeled.zero_()
        target.fill_(float("nan"))
    elif kind == "zero_scores":
        weights.zero_()
    elif kind == "perfect":
        target.fill_(4.0)
    loss = PoseOKSLoss([0.1, 0.1])(predicted, target, labeled, torch.full((count,), 1000.0), weights)
    assert loss.dtype == torch.float32
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.count_nonzero() == 0


@pytest.mark.parametrize("sigmas", [[], [0.0], [-0.1], [math.nan], [math.inf]])
def test_pose_oks_rejects_invalid_sigmas(sigmas):
    with pytest.raises(ValueError, match="sigma"):
        PoseOKSLoss(sigmas)
