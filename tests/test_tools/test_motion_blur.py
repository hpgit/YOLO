import random

import numpy as np
import pytest
import torch
from PIL import Image

from yolo.tools.motion_blur import MotionBlur
from yolo.tools.yolov9_augmentation import YOLOv9Augmentation


@pytest.mark.parametrize("direction", range(4))
@pytest.mark.parametrize("pil_input", [False, True])
def test_centered_blur_is_mild_and_preserves_labels(direction, pil_input, monkeypatch):
    pixels = np.zeros((9, 9, 3), dtype=np.uint8)
    pixels[4, 4] = 240
    image = Image.fromarray(pixels) if pil_input else pixels
    boxes = torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]])
    original_boxes = boxes.clone()
    monkeypatch.setattr(random, "randrange", lambda _: direction)
    output, labels = MotionBlur(prob=1)(image, boxes)
    result = np.asarray(output)

    assert isinstance(output, type(image))
    assert result.shape == pixels.shape and result.dtype == np.uint8
    assert result[4, 4, 0] == 160  # Retain 2/3 of the central impulse.
    assert np.count_nonzero(result[:, :, 0]) == 3
    np.testing.assert_array_equal(result, result[::-1, ::-1])
    assert result[:, :, 0].sum() == 240
    assert labels is boxes and torch.equal(boxes, original_boxes)
    np.testing.assert_array_equal(np.asarray(image), pixels)


@pytest.mark.parametrize("kwargs", [{"prob": 0}, {"strength": 0}])
def test_disabled_blur_is_identity_without_rng_consumption(kwargs):
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    boxes = np.zeros((0, 5), dtype=np.float32)
    state = random.getstate()
    output, labels = MotionBlur(**kwargs)(image, boxes)
    assert output is image and labels is boxes
    assert random.getstate() == state


def test_blur_preserves_constant_color_and_empty_labels_at_borders():
    image = np.full((7, 9, 3), (21, 80, 192), dtype=np.uint8)
    boxes = np.zeros((0, 5), dtype=np.float32)
    output, labels = MotionBlur(prob=1)(image, boxes)
    np.testing.assert_array_equal(output, image)
    assert labels is boxes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prob": -0.1},
        {"prob": 1.1},
        {"prob": float("nan")},
        {"strength": -0.1},
        {"strength": 1.1},
        {"strength": float("nan")},
        {"kernel_size": 1},
        {"kernel_size": 4},
        {"kernel_size": 3.5},
        {"kernel_size": True},
    ],
)
def test_invalid_settings_fail_early(kwargs):
    with pytest.raises(ValueError, match="motion blur"):
        MotionBlur(**kwargs)


def test_yolov9_blurs_pixels_after_geometry_without_changing_targets():
    image = np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    boxes = np.array([[0, 0.1, 0.2, 0.9, 0.8]], dtype=np.float32)

    def run(prob):
        random.seed(41)
        np.random.seed(41)
        augment = YOLOv9Augmentation(64, albumentations=False, mosaic=0, scale=0.1, motion_blur=prob)
        return augment(image, boxes)

    baseline, blurred, repeated = run(0), run(1), run(1)
    assert not torch.equal(baseline[0], blurred[0])
    assert torch.equal(baseline[1], blurred[1])
    assert torch.equal(baseline[2], blurred[2])
    for actual, expected in zip(blurred, repeated):
        assert torch.equal(actual, expected)
    assert torch.isfinite(blurred[0]).all()
    assert 0 <= blurred[0].min() <= blurred[0].max() <= 1
