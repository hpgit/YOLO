import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from yolo.tools.pose_dataset import (
    CocoPoseDataset,
    create_pose_dataloader,
    pose_collate_fn,
)

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def make_keypoints(*points):
    keypoints = [0.0] * 51
    for index, x, y, visibility in points:
        keypoints[index * 3 : index * 3 + 3] = [x, y, visibility]
    return keypoints


def write_pose_data(tmp_path, *, image_count=2, annotations=None):
    root = tmp_path / "coco"
    image_dir = root / "images" / "train2017"
    image_dir.mkdir(parents=True)
    images = []
    for image_id in range(1, image_count + 1):
        file_name = f"{image_id:012d}.jpg"
        # Asymmetric pixels make an accidental in-place/flip error observable.
        image = Image.new("RGB", (10, 20), (image_id, 0, 0))
        image.putpixel((0, 0), (255, 0, 0))
        image.save(image_dir / file_name)
        images.append({"id": image_id, "file_name": file_name, "width": 10, "height": 20})
    payload = {
        "images": images,
        "annotations": annotations or [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "keypoints": KEYPOINT_NAMES,
                "skeleton": [[16, 14], [14, 12]],
            }
        ],
    }
    annotation_dir = root / "annotations"
    annotation_dir.mkdir()
    (annotation_dir / "person_keypoints_train2017.json").write_text(json.dumps(payload))
    return root


def configs(root, *, fliplr=0.0, fraction=1.0, workers=0):
    data_cfg = SimpleNamespace(
        image_size=[20, 20],
        data_augment={"Pose": {"fliplr": fliplr}} if fliplr else {},
        dynamic_shape=False,
        batch_size=2,
        shuffle=False,
        cpu_num=workers,
        pin_memory=False,
    )
    dataset_cfg = {
        "path": str(root),
        "train": "train2017",
        "validation": "train2017",
        "class_num": 1,
        "num_keypoints": 17,
        "fraction": {"train": fraction, "validation": fraction},
        "subset_seed": 10,
    }
    return data_cfg, dataset_cfg


def person_annotations():
    keypoints = make_keypoints((0, 2, 4, 2), (1, 1, 5, 2), (2, 7, 6, 1), (3, 999, 999, 0))
    return [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [1, 2, 4, 8],
            "area": 32,
            "iscrowd": 0,
            "num_keypoints": 3,
            "keypoints": keypoints,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 2, 2],
            "area": 4,
            "iscrowd": 0,
            "num_keypoints": 0,
            "keypoints": [0] * 51,
        },
        {
            "id": 3,
            "image_id": 1,
            "category_id": 1,
            "bbox": [2, 2, 3, 3],
            "area": 9,
            "iscrowd": 1,
            "num_keypoints": 0,
            "keypoints": [0] * 51,
        },
    ]


def test_pose_targets_letterbox_empty_images_and_detection_only_people(tmp_path):
    root = write_pose_data(tmp_path, annotations=person_annotations())
    data_cfg, dataset_cfg = configs(root)
    dataset = CocoPoseDataset(data_cfg, dataset_cfg, "train")

    assert dataset.image_ids == [1, 2]
    image, targets, reverse, path = dataset[0]
    assert image.shape == (3, 20, 20)
    assert targets.shape == (2, 56)  # crowd skipped; no-keypoint person retained
    torch.testing.assert_close(targets[0, :5], torch.tensor([0.0, 6.0, 2.0, 10.0, 10.0]))
    keypoints = targets[0, 5:].reshape(17, 3)
    torch.testing.assert_close(keypoints[0], torch.tensor([7.0, 4.0, 2.0]))
    torch.testing.assert_close(keypoints[3], torch.zeros(3))  # v=0 coordinates sanitized
    assert not targets[1, 5:].any()  # box-only person still supervises detection
    torch.testing.assert_close(reverse, torch.tensor([1.0, 1.0, 5.0, 0.0]))
    assert Path(path).name == "000000000001.jpg"
    assert dataset[1][1].shape == (0, 56)  # empty split image is preserved


def test_pose_horizontal_flip_swaps_coco_left_and_right(tmp_path):
    root = write_pose_data(tmp_path, annotations=person_annotations())
    data_cfg, dataset_cfg = configs(root, fliplr=1.0)
    _, targets, _, _ = CocoPoseDataset(data_cfg, dataset_cfg, "train")[0]

    torch.testing.assert_close(targets[0, 1:5], torch.tensor([10.0, 2.0, 14.0, 10.0]))
    keypoints = targets[0, 5:].reshape(17, 3)
    torch.testing.assert_close(keypoints[0], torch.tensor([13.0, 4.0, 2.0]))
    # Reflect first, then swap semantic left/right indices.
    torch.testing.assert_close(keypoints[1], torch.tensor([8.0, 6.0, 1.0]))
    torch.testing.assert_close(keypoints[2], torch.tensor([14.0, 5.0, 2.0]))
    assert not targets[1, 5:].any()


def test_pose_collate_and_loader_keep_standard_batch_contract(tmp_path):
    root = write_pose_data(tmp_path, annotations=person_annotations())
    data_cfg, dataset_cfg = configs(root)
    dataset = CocoPoseDataset(data_cfg, dataset_cfg, "train")
    batch = pose_collate_fn([dataset[0], dataset[1]])
    assert batch[0] == 2
    assert batch[1].shape == (2, 3, 20, 20)
    assert batch[2].shape == (2, 2, 56)
    assert torch.all(batch[2][1, :, 0] == -1)
    assert batch[3].shape == (2, 4)
    assert len(batch[4]) == 2

    loader = create_pose_dataloader(data_cfg, dataset_cfg, "train")
    loaded = next(iter(loader))
    assert loaded[1].shape == batch[1].shape
    assert loaded[2].shape == batch[2].shape


def test_fraction_samples_all_image_ids_deterministically_and_records_manifest(tmp_path):
    root = write_pose_data(tmp_path, image_count=20)
    data_cfg, dataset_cfg = configs(root, fraction=0.05)
    first = CocoPoseDataset(data_cfg, dataset_cfg, "train")
    second = CocoPoseDataset(data_cfg, dataset_cfg, "train")

    assert first.image_ids == second.image_ids
    assert len(first) == 1
    assert first.records[0]["targets"].shape == (0, 56)
    assert first.manifest["total_images"] == 20
    assert first.manifest["selected_images"] == 1
    assert len(first.manifest["image_ids_sha256"]) == 64


@pytest.mark.parametrize(
    "update, match",
    [
        ({"bbox": [1, 2, float("nan"), 8]}, "non-finite"),
        ({"bbox": [8, 2, 4, 8]}, "outside"),
        ({"keypoints": make_keypoints((0, 11, 4, 2)), "num_keypoints": 1}, "outside"),
        ({"keypoints": make_keypoints((0, 2, 4, 3)), "num_keypoints": 1}, "visibility"),
        ({"keypoints": make_keypoints((0, 2, 4, 2)), "num_keypoints": 0}, "does not match"),
    ],
)
def test_invalid_pose_annotations_fail_before_training(tmp_path, update, match):
    annotation = person_annotations()[0]
    annotation.update(update)
    root = write_pose_data(tmp_path, annotations=[annotation])
    data_cfg, dataset_cfg = configs(root)
    with pytest.raises(ValueError, match=match):
        CocoPoseDataset(data_cfg, dataset_cfg, "train")


def test_pose_rejects_bbox_only_or_validation_augmentation(tmp_path):
    root = write_pose_data(tmp_path)
    data_cfg, dataset_cfg = configs(root)
    data_cfg.data_augment = {"HorizontalFlip": 0.5}
    with pytest.raises(ValueError, match="only supports"):
        CocoPoseDataset(data_cfg, dataset_cfg, "train")

    data_cfg.data_augment = {"Pose": {"fliplr": 0.5}}
    with pytest.raises(ValueError, match="only supported for training"):
        CocoPoseDataset(data_cfg, dataset_cfg, "validation")


def test_pose_hydra_configs_compose():
    config_dir = Path(__file__).resolve().parents[2] / "yolo" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["task=train-pose", "dataset=coco-pose-smoke", "model=v9-t-pose"],
        )
    assert cfg.task.task == "train"
    assert cfg.task.data.data_augment.Pose.fliplr == 0.5
    assert cfg.task.validation.evaluator == "coco-pose"
    assert cfg.dataset.pose is True
    assert cfg.dataset.fraction.train == 0.05
    assert cfg.model.pose.num_keypoints == cfg.dataset.num_keypoints == 17
