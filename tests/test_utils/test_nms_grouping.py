import pytest
import torch
from torchvision.ops import batched_nms

from yolo.utils.nms_utils import grouped_batched_nms


@pytest.mark.parametrize("count", [0, 32, 1000, 1001, 1600])
def test_grouped_nms_matches_torchvision_with_ties(count):
    generator = torch.Generator().manual_seed(28)
    corners = torch.randint(0, 24, (count, 2), generator=generator).float()
    boxes = torch.cat((corners, corners + 12), dim=1)
    groups = torch.randint(0, 16, (count,), generator=generator) * 3
    # Deliberate score ties and overlapping boxes across independent groups.
    scores = torch.randint(0, 8, (count,), generator=generator).float() / 8
    expected = batched_nms(boxes, scores, groups, 0.5)
    actual = grouped_batched_nms(boxes, scores, groups, 0.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_large_single_group_preserves_nms_order():
    boxes = torch.tensor([[0., 0., 10., 10.]]).repeat(1100, 1)
    scores = torch.ones(1100)
    groups = torch.full((1100,), 95, dtype=torch.long)
    torch.testing.assert_close(grouped_batched_nms(boxes, scores, groups, .7),
                               batched_nms(boxes, scores, groups, .7), rtol=0, atol=0)


def test_torchscript_fallback_accepts_float_threshold():
    compiled = torch.jit.script(grouped_batched_nms)
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]])
    scores = torch.tensor([.8, .9])
    groups = torch.tensor([0, 0])
    torch.testing.assert_close(compiled(boxes, scores, groups, .5),
                               batched_nms(boxes, scores, groups, .5), rtol=0, atol=0)
