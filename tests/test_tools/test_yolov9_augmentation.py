import copy
import random

import numpy as np
import pytest
import torch

from yolo.tools.yolov9_augmentation import (
    YOLOv9Augmentation,
    _copy_paste,
    _random_perspective,
)


def _disabled_hyp(**overrides):
    hyp = {
        "albumentations": False,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
    }
    hyp.update(overrides)
    return hyp


class _SamplePool:
    def __init__(self, samples):
        self.samples = samples

    def __call__(self, count=1):
        if count == 1:
            return self.samples[random.randint(0, len(self.samples) - 1)]
        return random.choices(self.samples, k=count)


def _sample(value, class_id):
    image = np.full((40, 64, 3), value, dtype=np.uint8)
    image[8:25, 10:30] = (value, min(value + 20, 255), max(value - 20, 0))
    boxes = np.array([[class_id, 0.15, 0.2, 0.5, 0.7]], dtype=np.float32)
    segments = [
        np.array([[0.15, 0.2], [0.5, 0.2], [0.5, 0.7], [0.15, 0.7]], dtype=np.float32)
    ]
    return image, boxes, segments


def test_fixed_seed_is_repeatable_and_does_not_mutate_samples():
    primary = _sample(50, 0)
    pool_samples = [_sample(80, 1), _sample(120, 2), _sample(160, 3)]
    originals = copy.deepcopy((primary, pool_samples))
    augmentation = YOLOv9Augmentation(
        64,
        **_disabled_hyp(
            mosaic=1.0,
            mixup=1.0,
            copy_paste=1.0,
            degrees=5.0,
            translate=0.1,
            scale=0.2,
            shear=2.0,
            perspective=0.0005,
            hsv_h=0.015,
            hsv_s=0.3,
            hsv_v=0.2,
            flipud=0.5,
            fliplr=0.5,
        ),
    )

    results = []
    for _ in range(2):
        random.seed(321)
        np.random.seed(321)
        results.append(augmentation(*primary, get_sample=_SamplePool(pool_samples)))

    for first, second in zip(results[0], results[1]):
        assert torch.equal(first, second)

    for actual, original in zip(primary[:2], originals[0][:2]):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(primary[2][0], originals[0][2][0])
    for actual_sample, original_sample in zip(pool_samples, originals[1]):
        np.testing.assert_array_equal(actual_sample[0], original_sample[0])
        np.testing.assert_array_equal(actual_sample[1], original_sample[1])
        np.testing.assert_array_equal(actual_sample[2][0], original_sample[2][0])


def test_no_mosaic_identity_geometry_only_adds_expected_letterbox_padding():
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 20
    image[:, :, 1] = 80
    image[:, :, 2] = 160
    boxes = np.array([[4, 0.25, 0.25, 0.75, 0.75]], dtype=np.float32)
    augmentation = YOLOv9Augmentation(64, **_disabled_hyp())

    output, output_boxes, reverse = augmentation(image, boxes)

    expected_image = np.full((64, 64, 3), 114, dtype=np.uint8)
    expected_image[16:48] = image
    expected_image = torch.from_numpy(expected_image.transpose(2, 0, 1).copy()).float() / 255
    expected_boxes = torch.tensor([[4, 0.25, 0.375, 0.75, 0.625]], dtype=torch.float32)
    assert torch.equal(output, expected_image)
    assert torch.allclose(output_boxes, expected_boxes)
    assert torch.equal(reverse, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))


def test_mixed_polygon_and_empty_segment_keep_box_and_only_copy_real_polygon():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[8:19, 3:13] = (25, 100, 225)
    labels = np.array(
        [[3, 3, 8, 12, 18], [7, 25, 8, 35, 18]],
        dtype=np.float32,
    )
    segments = [
        np.array([[3, 8], [12, 8], [12, 18], [3, 18]], dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
    ]

    random.seed(9)
    warped, warped_labels = _random_perspective(
        image.copy(), labels.copy(), segments, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    assert warped_labels[:, 0].tolist() == [3, 7]
    np.testing.assert_allclose(warped_labels[:, 1:], labels[:, 1:], atol=1e-5)

    random.seed(9)
    pasted, pasted_labels, pasted_segments = _copy_paste(
        warped, warped_labels, [segment.copy() for segment in segments], 1.0
    )
    assert pasted_labels[:, 0].tolist() == [3, 7, 3]
    np.testing.assert_allclose(pasted_labels[-1, 1:], [52, 8, 61, 18])
    assert len(pasted_segments) == 3
    assert pasted_segments[1].shape == (0, 2)
    np.testing.assert_array_equal(pasted[10, 55], image[10, 8])


def test_empty_labels_support_mosaic_mixup_and_albumentations():
    image = np.full((24, 32, 3), (10, 60, 180), dtype=np.uint8)
    empty = np.zeros((0, 5), dtype=np.float32)
    pool = _SamplePool([(image.copy(), empty.copy(), []) for _ in range(3)])
    augmentation = YOLOv9Augmentation(
        32,
        **_disabled_hyp(albumentations=True, mosaic=1.0, mixup=1.0),
    )

    random.seed(12)
    np.random.seed(12)
    output, boxes, reverse = augmentation(image, empty, [], pool)

    assert output.shape == (3, 32, 32)
    assert output.dtype == torch.float32
    assert boxes.shape == (0, 5)
    assert reverse.shape == (5,)


def test_forced_flips_transform_normalized_xyxy():
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    boxes = np.array([[2, 0.1, 0.2, 0.7, 0.6]], dtype=np.float32)
    augmentation = YOLOv9Augmentation(
        16, **_disabled_hyp(flipud=1.0, fliplr=1.0)
    )

    output, output_boxes, _ = augmentation(image, boxes)

    expected_image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255
    assert torch.equal(output, torch.flip(expected_image, dims=(1, 2)))
    assert torch.allclose(
        output_boxes,
        torch.tensor([[2, 0.3, 0.4, 0.9, 0.8]], dtype=torch.float32),
    )


@pytest.mark.parametrize("image_size", [0, -2, 63, True, (64, 32), [64], None])
def test_invalid_image_sizes_are_rejected(image_size):
    with pytest.raises(ValueError):
        YOLOv9Augmentation(image_size, albumentations=False)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("mosaic", -0.01),
        ("mixup", 1.01),
        ("copy_paste", -1),
        ("flipud", 2),
        ("fliplr", -0.5),
    ],
)
def test_invalid_probabilities_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        YOLOv9Augmentation(64, albumentations=False, **{name: value})
