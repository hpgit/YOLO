import pytest
import torch

from yolo.tools.loss_functions import BoxLoss, DualLoss
from yolo.utils.bounding_box_utils import calculate_iou


def _legacy_box_loss(predicts_bbox, targets_bbox, valid_masks, box_norm, cls_norm):
    valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
    picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
    picked_targets = targets_bbox[valid_bbox].view(-1, 4)
    iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
    return ((1.0 - iou) * box_norm).sum() / cls_norm


@pytest.mark.parametrize("metrics", ["iou", "diou", "ciou"])
def test_aligned_iou_matches_pairwise_diagonal_values_and_gradients(metrics):
    boxes1 = torch.tensor(
        [[0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 4.0, 5.0], [2.0, 2.0, 2.0, 6.0]],
        requires_grad=True,
    )
    boxes2 = torch.tensor(
        [[0.5, 0.25, 2.5, 2.75], [0.0, 2.0, 3.5, 5.5], [2.0, 1.0, 2.0, 7.0]],
    )

    pairwise = calculate_iou(boxes1, boxes2, metrics).diag()
    pairwise.sum().backward()
    pairwise_grad = boxes1.grad.detach().clone()

    boxes1.grad = None
    aligned = calculate_iou(boxes1, boxes2, metrics, aligned=True)
    aligned.sum().backward()

    torch.testing.assert_close(aligned, pairwise, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(boxes1.grad, pairwise_grad, rtol=0, atol=0, equal_nan=True)


def test_aligned_iou_supports_batched_and_empty_boxes_with_float32_numerics():
    boxes1 = torch.tensor(
        [[[0, 0, 2, 2], [1, 1, 3, 4]], [[2, 2, 5, 6], [0, 1, 0, 3]]], dtype=torch.float64
    )
    boxes2 = torch.tensor(
        [[[1, 0, 3, 2], [0, 2, 4, 5]], [[1, 3, 6, 7], [0, 0, 0, 4]]], dtype=torch.float64
    )

    pairwise = calculate_iou(boxes1, boxes2, "ciou")
    aligned = calculate_iou(boxes1, boxes2, "ciou", aligned=True)

    assert aligned.shape == (2, 2)
    assert aligned.dtype == torch.float64
    torch.testing.assert_close(aligned, pairwise.diagonal(dim1=-2, dim2=-1), rtol=0, atol=0, equal_nan=True)

    empty = torch.empty((0, 4), dtype=torch.float32)
    empty_aligned = calculate_iou(empty, empty, "ciou", aligned=True)
    assert empty_aligned.shape == (0,)
    assert empty_aligned.dtype == torch.float32


def test_aligned_iou_requires_identical_shapes():
    with pytest.raises(ValueError, match="same shape"):
        calculate_iou(torch.empty(2, 4), torch.empty(3, 4), aligned=True)


def test_pairwise_iou_behavior_is_preserved():
    boxes1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 4.0]])
    boxes2 = torch.tensor([[1.0, 0.0, 3.0, 2.0], [0.0, 2.0, 4.0, 5.0], [4.0, 4.0, 5.0, 5.0]])

    actual = calculate_iou(boxes1, boxes2)
    expected = torch.tensor([[1.0 / 3.0, 0.0, 0.0], [1.0 / 4.0, 2.0 / 7.0, 0.0]])

    assert actual.shape == (2, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("targets", "valid_masks", "box_norm"),
    [
        (
            torch.tensor([[[0.5, 0.5, 2.5, 3.5], [1.0, 1.0, 4.0, 5.0], [2.0, 1.0, 2.0, 6.0]]]),
            torch.tensor([[True, True, True]]),
            torch.tensor([1.0, 0.5, 2.0]),
        ),
        (
            torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 3.0], [2.0, 2.0, 4.0, 2.0]]]),
            torch.tensor([[True, True, True]]),
            torch.tensor([1.0, 1.0, 1.0]),
        ),
        (
            torch.zeros(1, 3, 4),
            torch.tensor([[False, False, False]]),
            torch.empty(0),
        ),
    ],
    ids=["regular", "degenerate", "empty"],
)
def test_box_loss_matches_pairwise_diagonal_value_and_gradient(targets, valid_masks, box_norm):
    predicts = torch.tensor(
        [[[0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 4.5, 5.5], [2.0, 2.0, 2.0, 6.0]]],
        requires_grad=True,
    )
    cls_norm = torch.tensor(3.5)

    legacy_loss = _legacy_box_loss(predicts, targets, valid_masks, box_norm, cls_norm)
    legacy_loss.backward()
    legacy_grad = predicts.grad.detach().clone()

    predicts.grad = None
    optimized_loss = BoxLoss()(predicts, targets, valid_masks, box_norm, cls_norm)
    optimized_loss.backward()

    torch.testing.assert_close(optimized_loss, legacy_loss, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(predicts.grad, legacy_grad, rtol=0, atol=0, equal_nan=True)


def test_dual_loss_logging_keeps_detached_tensors_without_changing_gradients():
    class EchoLoss:
        def __call__(self, predicts, targets):
            return tuple(predicts)

    dual_loss = DualLoss.__new__(DualLoss)
    dual_loss.loss = EchoLoss()
    dual_loss.aux_rate = 0.5
    dual_loss.iou_rate = 1.0
    dual_loss.dfl_rate = 2.0
    dual_loss.cls_rate = 3.0

    auxiliary = [torch.tensor(value, requires_grad=True) for value in (1.0, 2.0, 3.0)]
    main = [torch.tensor(value, requires_grad=True) for value in (4.0, 5.0, 6.0)]
    total, logged = dual_loss(auxiliary, main, torch.empty(0))

    expected = torch.tensor([4.5, 12.0, 22.5])
    torch.testing.assert_close(torch.stack(list(logged.values())), expected)
    assert all(isinstance(value, torch.Tensor) and not value.requires_grad for value in logged.values())

    total.backward()
    torch.testing.assert_close(torch.stack([value.grad for value in auxiliary]), torch.tensor([0.5, 1.0, 1.5]))
    torch.testing.assert_close(torch.stack([value.grad for value in main]), torch.tensor([1.0, 2.0, 3.0]))
