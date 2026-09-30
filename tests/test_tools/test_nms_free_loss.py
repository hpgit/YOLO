from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from yolo.tools.loss_functions import (
    DualLoss,
    NMSFreeLoss,
    OneToOneMatcher,
    create_loss_function,
)
from yolo.utils.bounding_box_utils import BoxMatcher


def _converter():
    return SimpleNamespace(
        anchor_grid=torch.tensor([[4, 4], [12, 4], [20, 4], [4, 12], [12, 12], [20, 12]]),
        scaler=torch.full((6,), 8),
    )


def _config(nms_free=True, one2one=1.0):
    return OmegaConf.create(
        {
            "model": {"nms_free": nms_free, "anchor": {"reg_max": 4}},
            "dataset": {"class_num": 3},
            "task": {
                "loss": {
                    "objective": {"BoxLoss": 7.5, "DFLoss": 1.5, "BCELoss": 0.5},
                    "aux": 0.25,
                    "one2one": one2one,
                    "matcher": {"iou": "CIoU", "topk": 2, "factor": {"iou": 6.0, "cls": 0.5}},
                }
            },
        }
    )


def _matcher():
    cfg = _config()
    return OneToOneMatcher(cfg.task.loss.matcher, 3, _converter(), reg_max=4)


def _predicts(batch=1):
    converter = _converter()
    cls = torch.zeros((batch, 6, 3), requires_grad=True)
    dist = torch.zeros((batch, 6, 4, 4), requires_grad=True)
    centers = converter.anchor_grid.float()
    boxes = torch.cat((centers - 5, centers + 5), -1).repeat(batch, 1, 1).requires_grad_()
    return cls, dist, boxes


def test_top1_nominates_at_most_one_anchor_per_gt_and_has_no_gt_collisions():
    matcher = _matcher()
    targets = torch.tensor([[[0, 0, 0, 16, 16], [1, 16, 0, 24, 16]]], dtype=torch.float32)
    predicts = _predicts()
    matched, valid = matcher(targets, (predicts[0], predicts[2]))
    assert matcher.topk == 1
    assert valid.sum() == 2
    for target in targets[0]:
        assert ((matched[0, :, -4:] == target[1:]).all(-1) & valid[0]).sum() == 1
    assert (matched[..., :3] > 0).sum(-1).max() == 1
    assert not matched.requires_grad


def test_colliding_gt_and_equal_anchor_scores_have_deterministic_single_winner():
    matcher = _matcher()
    targets = torch.tensor([[[0, 0, 0, 16, 16], [1, 0, 0, 16, 16]]], dtype=torch.float32)
    cls = torch.zeros(1, 6, 3)
    boxes = targets[:, :1, 1:].expand(-1, 6, -1)
    matched, valid = matcher(targets, (cls, boxes))
    assert valid.tolist() == [[True, False, False, False, False, False]]
    torch.testing.assert_close(matched[0, 0, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.count_nonzero(matched[~valid]) == 0


def test_collision_winner_must_be_one_of_the_nominating_targets(monkeypatch):
    matcher = _matcher()
    targets = torch.tensor([[[0, 0, 0, 24, 24], [1, 0, 0, 24, 24], [2, 0, 0, 24, 24]]]).float()
    # GT 2 has the best IoU at anchor 0, but only nominates anchor 1.
    ious = torch.tensor([[[0.8, 0.1, 0, 0, 0, 0], [0.7, 0.1, 0, 0, 0, 0], [0.9, 1, 0, 0, 0, 0]]])
    monkeypatch.setattr(matcher, "get_iou_matrix", lambda *args: ious)
    matched, valid = matcher(targets, (torch.zeros(1, 6, 3), torch.zeros(1, 6, 4)))
    assert valid.tolist() == [[True, True, False, False, False, False]]
    assert matched[0, 0, :3].argmax() == 0
    assert matched[0, 1, :3].argmax() == 2


@pytest.mark.parametrize("target_count", [0, 3])
def test_empty_and_all_padded_batches_are_true_background(target_count):
    targets = torch.full((2, target_count, 5), -1.0)
    cls, _, boxes = _predicts(batch=2)
    matched, valid = _matcher()(targets, (cls, boxes))
    assert matched.shape == (2, 6, 7)
    assert not valid.any()
    assert torch.count_nonzero(matched) == 0


def test_padding_and_invalid_gt_rows_cannot_enter_fallback_matching():
    targets = torch.tensor(
        [
            [[0, 0, 0, 16, 16], [-1, 0, 0, 16, 16], [3, 0, 0, 16, 16], [0, 0, 0, 0, 16]],
            [[-1, 0, 0, 16, 16], [0.5, 0, 0, 16, 16], [0, 0, 0, float("nan"), 16], [-1, -1, -1, -1, -1]],
        ]
    )
    cls, _, boxes = _predicts(batch=2)
    matched, valid = _matcher()(targets, (cls, boxes))
    assert valid.sum(-1).tolist() == [1, 0]
    assert torch.isfinite(matched).all()
    assert torch.count_nonzero(matched[~valid]) == 0


def test_float16_alignment_powers_do_not_underflow_and_targets_stay_float32():
    matcher = _matcher()
    matcher.iou = "iou"
    targets = torch.tensor([[[0, 0, 0, 8, 8]]], dtype=torch.float16)
    cls = torch.full((1, 6, 3), -10.0, dtype=torch.float16, requires_grad=True)
    boxes = torch.tensor([[[3, 3, 5, 5]]], dtype=torch.float16).expand(-1, 6, -1).requires_grad_()
    matched, valid = matcher(targets, (cls, boxes))
    assert valid.sum() == 1
    assert matched.dtype == torch.float32
    torch.testing.assert_close(matched[0, 0, 0], torch.tensor(1 / 16))
    assert not matched.requires_grad


def test_small_gt_without_an_inside_anchor_uses_single_fallback():
    targets = torch.tensor([[[0, 5, 5, 6, 6]]], dtype=torch.float32)
    matcher = _matcher()
    matcher.iou = "iou"
    cls, _, boxes = _predicts()
    assert not matcher.get_valid_matrix(targets[..., 1:]).any()
    matched, valid = matcher(targets, (cls, boxes))
    assert valid.sum() == 1
    torch.testing.assert_close(matched[valid][:, -4:], targets[0, :, 1:])


@pytest.mark.parametrize("with_aux", [False, True])
@pytest.mark.parametrize("with_targets", [False, True])
def test_dual_branch_loss_is_additive_and_produces_finite_gradients(with_aux, with_targets):
    cfg = _config(one2one=0.7)
    converter = _converter()
    criterion = create_loss_function(cfg, converter)
    targets = torch.tensor([[[0, 0, 0, 16, 16]]]).float() if with_targets else torch.empty(1, 0, 5)
    aux = _predicts() if with_aux else None
    main, one2one = _predicts(), _predicts()
    total, metrics = criterion(aux, main, targets, one2one_predicts=one2one)
    dense, _ = DualLoss(cfg, converter)(aux, main, targets)
    o2o_terms = criterion.one2one_loss(one2one, targets)
    expected_o2o = 0.7 * sum(rate * term for rate, term in zip((7.5, 1.5, 0.5), o2o_terms))
    torch.testing.assert_close(total, dense + expected_o2o)
    torch.testing.assert_close(metrics["Loss/One2OneLoss"], expected_o2o)
    torch.testing.assert_close(metrics["Loss/One2ManyLoss"], dense)
    torch.testing.assert_close(sum(metrics[f"Loss/{name}Loss"] for name in ("Box", "DFL", "BCE")), total)
    assert all(not value.requires_grad for value in metrics.values())
    total.backward()
    for branch in [main, one2one, aux] if with_aux else [main, one2one]:
        for prediction in branch:
            assert prediction.grad is not None
            assert torch.isfinite(prediction.grad).all()
        assert branch[0].grad.abs().sum() > 0
        if with_targets:
            assert branch[1].grad.abs().sum() > 0
            assert branch[2].grad.abs().sum() > 0


def test_nms_free_requires_an_independent_one2one_prediction():
    criterion = create_loss_function(_config(), _converter())
    with pytest.raises(ValueError, match="one2one_predicts"):
        criterion(None, _predicts(), torch.empty(1, 0, 5))


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_one2one_loss_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="finite and positive"):
        create_loss_function(_config(one2one=weight), _converter())


def test_opt_out_preserves_original_dual_loss_and_matcher():
    cfg = _config(nms_free=False)
    criterion = create_loss_function(cfg, _converter())
    assert type(criterion) is DualLoss
    assert type(criterion.loss.matcher) is BoxMatcher
    assert criterion.loss.matcher.topk == cfg.task.loss.matcher.topk
    del cfg.model.nms_free
    assert type(create_loss_function(cfg, _converter())) is DualLoss
