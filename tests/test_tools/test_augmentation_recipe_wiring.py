"""Training recipe integration and target preservation contracts."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from yolo.tools.data_loader import collate_fn, create_dataloader


def _config():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "yolo/config"), version_base=None):
        return compose(config_name="config", overrides=["task=train", "dataset=mock", "cpu_num=0"])


def test_mosaic_collation_preserves_more_than_100_targets():
    labels = torch.zeros(137, 5)
    labels[:, 0] = torch.arange(137)
    batch = [
        (torch.zeros(3, 32, 32), labels, torch.tensor([1, 0, 0, 0, 0]), Path("a.jpg")),
        (torch.zeros(3, 32, 32), torch.zeros(0, 5), torch.tensor([1, 0, 0, 0, 0]), Path("b.jpg")),
    ]
    _, _, targets, _, _ = collate_fn(batch, max_boxes=None)
    assert targets.shape == (2, 137, 5)
    torch.testing.assert_close(targets[0], labels, rtol=0, atol=0)
    assert (targets[1, :, 0] == -1).all()


def test_recipe_cannot_be_applied_to_validation_or_combined_with_legacy_transforms():
    cfg = _config()
    cfg.dataset.auto_download = None
    with pytest.raises(ValueError, match="only supported for training"):
        create_dataloader(cfg.task.data, cfg.dataset, "validation")
    from omegaconf import open_dict

    with open_dict(cfg.task.data.data_augment):
        cfg.task.data.data_augment.HorizontalFlip = 0.5
    with pytest.raises(ValueError, match="complete augmentation recipe"):
        create_dataloader(cfg.task.data, cfg.dataset, "train")


def test_default_recipe_uses_polygon_dataset_and_uncapped_collation(tmp_path):
    from yolo.tools.yolov9_dataset import YOLOv9Dataset

    image_path = tmp_path / "images" / "train" / "polygon.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(64, 128, 192)).save(image_path)
    label_path = tmp_path / "labels" / "train" / "polygon.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")

    cfg = _config()
    cfg.dataset.path = str(tmp_path)
    cfg.dataset.auto_download = None
    blur = cfg.task.data.data_augment.YOLOv9
    assert (blur.motion_blur, blur.motion_blur_kernel_size, blur.motion_blur_strength) == (0.1, 3, 0.5)
    loader = create_dataloader(cfg.task.data, cfg.dataset, "train")
    assert isinstance(loader.dataset, YOLOv9Dataset)
    assert loader.dataset.img_paths == [image_path]
    assert loader.dataset.segments[0][0].shape == (4, 2)
    assert loader.collate_fn.keywords["max_boxes"] is None
    sample = loader.dataset[0]
    assert sample[0].shape == (3, 640, 640)
    assert sample[1].shape[1] == 5
    assert torch.isfinite(sample[0]).all() and torch.isfinite(sample[1]).all()
    assert sample[2].shape == (5,)
