import pytest
import torch
from PIL import Image

from tests.test_tools.test_pose_dataset import (
    configs,
    person_annotations,
    write_pose_data,
)
from yolo.tools.pose_augmentation import pose_affine
from yolo.tools.pose_dataset import CocoPoseDataset, create_pose_dataloader


def targets_for_transform():
    targets = torch.zeros(1, 59)
    targets[0, :5] = torch.tensor([0.0, 10.0, 20.0, 40.0, 60.0])
    targets[0, 5:11] = torch.tensor([20.5, 30.5, 2, 30.0, 40.0, 1])
    targets[0, -3:] = torch.tensor([600.0, 1, 1])
    return targets


def test_affine_rotation_aligns_pixels_boxes_and_both_labeled_visibilities():
    image = Image.new("RGB", (100, 100))
    image.putpixel((20, 30), (255, 0, 0))
    image, targets = pose_affine(image, targets_for_transform(), degrees=90, translate=(0, 0), scale=1)
    # Floating-point cos(pi/2) can lose one intensity level in interpolation.
    assert image.getpixel((69, 20))[0] >= 254
    torch.testing.assert_close(targets[0, 1:5], torch.tensor([40.0, 10.0, 80.0, 40.0]))
    torch.testing.assert_close(targets[0, 5:11], torch.tensor([69.5, 20.5, 2, 60.0, 30.0, 1]))
    assert not targets[0, 11:56].any()
    torch.testing.assert_close(targets[0, -3:], torch.tensor([600.0, 1, 1]))


def test_affine_scale_preserves_instance_area_instead_of_axis_aligned_box_area():
    _, targets = pose_affine(
        Image.new("RGB", (100, 100)), targets_for_transform(), degrees=30, translate=(0, 0), scale=0.5
    )
    torch.testing.assert_close(targets[0, -3:], torch.tensor([150.0, 1, 1]))
    box_area = (targets[0, 3:5] - targets[0, 1:3]).prod()
    assert box_area != targets[0, -3]


def test_affine_cropping_masks_joints_filters_people_and_retains_full_instance_scale():
    targets = targets_for_transform().repeat(3, 1)
    targets[:, 1:5] = torch.tensor([[5, 20, 25, 60], [40, 20, 90, 60], [29, 20, 49, 60]])
    targets[0, 5:11] = torch.tensor([10, 30, 1, 40, 40, 2])
    _, targets = pose_affine(Image.new("RGB", (100, 100)), targets, degrees=0, translate=(70, 0), scale=1)
    assert targets.shape == (1, 59)
    torch.testing.assert_close(targets[0, 1:5], torch.tensor([75.0, 20.0, 95.0, 60.0]))
    torch.testing.assert_close(targets[0, 5:11], torch.tensor([80.0, 30.0, 1, 0, 0, 0]))
    assert targets[0, -3] == 600


def test_affine_empty_image_still_augments_and_preserves_empty_target_shape():
    image, targets = pose_affine(
        Image.new("RGB", (10, 20), "white"), torch.empty(0, 59), degrees=10, translate=(4, 0), scale=0.8
    )
    assert image.size == (10, 20) and image.getpixel((0, 0)) == (0, 0, 0)
    assert targets.shape == (0, 59)


@pytest.mark.parametrize("resize_mode, gains", [("letterbox", (1.5, 1.5)), ("stretch", (3.0, 1.5))])
def test_dataset_scales_annotation_area_and_keeps_exact_resize_gains(tmp_path, resize_mode, gains):
    annotation = person_annotations()[0]
    annotation["area"] = 17.0  # Deliberately unlike the 32-pixel bounding-box area.
    root = write_pose_data(tmp_path, annotations=[annotation])
    data_cfg, dataset_cfg = configs(root)
    data_cfg.image_size = [30, 30]
    data_cfg.resize_mode = resize_mode
    dataset = CocoPoseDataset(data_cfg, dataset_cfg, "train")
    _, targets, reverse, _ = dataset[0]
    torch.testing.assert_close(targets[0, -3:], torch.tensor([17 * gains[0] * gains[1], *gains]))
    torch.testing.assert_close(reverse[:2], torch.tensor(gains))
    torch.testing.assert_close(dataset.records[0]["targets"][0, -3:], torch.tensor([17.0, 1, 1]))


@pytest.mark.parametrize("key", ["fliplr", "degrees", "translate", "scale", "hsv_h", "hsv_s", "hsv_v"])
def test_each_pose_augmentation_rejects_validation_and_nonfinite_values(tmp_path, key):
    root = write_pose_data(tmp_path)
    data_cfg, dataset_cfg = configs(root)
    for value in (-1, float("nan"), True, "0.1"):
        data_cfg.data_augment = {"Pose": {key: value}}
        with pytest.raises(ValueError, match=key):
            CocoPoseDataset(data_cfg, dataset_cfg, "train")
    data_cfg.data_augment = {"Pose": {key: 0.1}}
    with pytest.raises(ValueError, match="only supported for training"):
        CocoPoseDataset(data_cfg, dataset_cfg, "validation")


@pytest.mark.parametrize("options", [{"hsv_h": 0.6}, {"scale": 1.0}, {"translate": 1.1}, {"shear": 10}])
def test_invalid_pose_augmentation_ranges_and_unknown_options_fail(tmp_path, options):
    root = write_pose_data(tmp_path)
    data_cfg, dataset_cfg = configs(root)
    data_cfg.data_augment = {"Pose": options}
    with pytest.raises(ValueError, match=next(iter(options))):
        CocoPoseDataset(data_cfg, dataset_cfg, "train")


def test_disabled_augmentation_preserves_pixels_targets_and_torch_rng(tmp_path):
    root = write_pose_data(tmp_path, annotations=person_annotations())
    data_cfg, dataset_cfg = configs(root)
    plain = CocoPoseDataset(data_cfg, dataset_cfg, "train")
    data_cfg.data_augment = {
        "Pose": dict.fromkeys(("fliplr", "degrees", "translate", "scale", "hsv_h", "hsv_s", "hsv_v"), 0)
    }
    disabled = CocoPoseDataset(data_cfg, dataset_cfg, "train")
    state = torch.random.get_rng_state()
    left, right = plain[0], disabled[0]
    for actual, expected in zip(left[:3], right[:3]):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, torch.random.get_rng_state())


def test_color_augmentation_changes_pixels_without_changing_targets(tmp_path):
    root = write_pose_data(tmp_path, annotations=person_annotations())
    data_cfg, dataset_cfg = configs(root)
    plain = CocoPoseDataset(data_cfg, dataset_cfg, "train")[0]
    data_cfg.data_augment = {"Pose": {"hsv_h": 0.1, "hsv_s": 0.4, "hsv_v": 0.4}}
    torch.manual_seed(42)
    augmented = CocoPoseDataset(data_cfg, dataset_cfg, "train")[0]
    assert not torch.equal(plain[0], augmented[0])
    torch.testing.assert_close(plain[1], augmented[1])
    torch.testing.assert_close(plain[2], augmented[2])


def test_augmented_workers_are_reproducible_and_do_not_mutate_cached_annotations(tmp_path):
    root = write_pose_data(tmp_path, annotations=person_annotations())
    data_cfg, dataset_cfg = configs(root, workers=2)
    data_cfg.data_augment = {
        "Pose": {
            "fliplr": 0.5,
            "degrees": 10,
            "translate": 0.1,
            "scale": 0.2,
            "hsv_h": 0.015,
            "hsv_s": 0.4,
            "hsv_v": 0.4,
        }
    }
    torch.manual_seed(42)
    first = create_pose_dataloader(data_cfg, dataset_cfg, "train")
    # Pytest imports Torch before worker startup; avoid forking its thread pools.
    first.multiprocessing_context = "spawn"
    first.timeout = 30
    original = first.dataset.records[0]["targets"].clone()
    batch1 = next(iter(first))
    torch.manual_seed(42)
    second = create_pose_dataloader(data_cfg, dataset_cfg, "train")
    second.multiprocessing_context = "spawn"
    second.timeout = 30
    batch2 = next(iter(second))
    torch.testing.assert_close(batch1[1], batch2[1])
    torch.testing.assert_close(batch1[2], batch2[2])
    torch.testing.assert_close(first.dataset.records[0]["targets"], original)
    first.dataset[0]
    torch.testing.assert_close(first.dataset.records[0]["targets"], original)
