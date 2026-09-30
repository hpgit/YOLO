"""Independent geometry, policy wiring and original-coordinate output contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tests.conftest import get_cfg
from yolo.tools.data_augmentation import PadAndResize
from yolo.tools.data_loader import StreamDataLoader, create_dataloader
from yolo.tools.yolov9_augmentation import YOLOv9Augmentation
from yolo.utils.model_utils import PostProcess, predicts_to_json
from yolo.utils.resize import restore_boxes


@pytest.mark.parametrize("source", [(101, 37), (37, 101), (80, 80)])
@pytest.mark.parametrize("target", [(64, 32), (64, 64)])
@pytest.mark.parametrize("empty", [False, True])
def test_stretch_pixels_normalized_labels_and_roundtrip(source, target, empty):
    image = Image.fromarray(np.random.default_rng(3).integers(0, 256, (source[1], source[0], 3), dtype=np.uint8))
    labels = torch.tensor([[0, 0.1, 0.2, 0.8, 0.9]], dtype=torch.float32)
    if empty:
        labels = labels[:0]
    original = labels.clone()
    output, actual, reverse = PadAndResize(target, resize_mode="stretch")(image, labels)
    np.testing.assert_array_equal(np.asarray(output), np.asarray(image.resize(target, Image.Resampling.LANCZOS)))
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    pixel_boxes = actual[:, 1:] * torch.tensor([*target, *target])
    restored = restore_boxes(pixel_boxes, reverse)
    torch.testing.assert_close(restored, original[:, 1:] * torch.tensor([*source, *source]))
    torch.testing.assert_close(restore_boxes(pixel_boxes[None], reverse[None])[0], restored)


def test_postprocess_and_json_restore_two_axis_scale():
    reverse = torch.tensor([[0.5, 2.0, 0, 0, 0, 0]])
    boxes = torch.tensor([[[5.0, 40.0, 40.0, 180.0]]])
    scores = torch.tensor([[[0.9]]])
    post = PostProcess(lambda _: (scores, None, boxes), SimpleNamespace(min_confidence=0.5, min_iou=0.5, max_bbox=10))
    result = post({"Main": None}, rev_tensor=reverse)[0]
    torch.testing.assert_close(result[:, 1:5], torch.tensor([[10.0, 20.0, 80.0, 90.0]]))
    raw = torch.tensor([[0, 5, 40, 40, 180, 0.9]])
    record = predicts_to_json(["123.jpg"], [raw], reverse)[0]
    assert record["bbox"] == pytest.approx([10, 20, 70, 70])
    torch.testing.assert_close(raw[:, 1:5], boxes[0])


@pytest.mark.parametrize("task", ["train", "validation", "inference", "export"])
def test_shared_hydra_setting(task):
    cfg = get_cfg([f"task={task}", "resize_mode=stretch"])
    if task != "export":
        assert cfg.task.data.resize_mode == "stretch"
    if task == "train":
        assert cfg.task.validation.data.resize_mode == "stretch"
    assert get_cfg([f"task={task}"]).resize_mode == "letterbox"


@pytest.mark.parametrize("legacy", [False, True])
def test_train_and_validation_loaders_receive_mode(tmp_path, legacy):
    from tests.test_tools.test_yolov9_augmentation import _disabled_hyp

    cfg = get_cfg(["task=train", "dataset=mock", "resize_mode=stretch", "image_size=[64,64]", "cpu_num=0"])
    cfg.dataset.path = str(tmp_path)
    cfg.dataset.auto_download = None
    cfg.dataset.validation = "validation"
    cfg.task.data.batch_size = cfg.task.validation.data.batch_size = 1
    cfg.task.data.data_augment = {} if legacy else {"YOLOv9": _disabled_hyp()}
    for phase, data in [("train", cfg.task.data), ("validation", cfg.task.validation.data)]:
        image_path = tmp_path / "images" / phase / "sample.png"
        label_path = tmp_path / "labels" / phase / "sample.txt"
        image_path.parent.mkdir(parents=True)
        label_path.parent.mkdir(parents=True)
        Image.new("RGB", (100, 40), (255, 0, 0)).save(image_path)
        label_path.write_text("0 0.5 0.5 0.5 0.5\n")
        loader = create_dataloader(data, cfg.dataset, phase)
        _, images, labels, _, _ = next(iter(loader))
        torch.testing.assert_close(images[0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(labels[0, 0, 1:], torch.tensor([16.0, 16.0, 48.0, 48.0]))


def test_stream_loader_receives_mode(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (100, 40), (255, 0, 0)).save(path)
    cfg = get_cfg(["task=inference", "resize_mode=stretch", "image_size=[64,64]", f"task.data.source={path}"])
    loader = StreamDataLoader(cfg.task.data)
    try:
        image, reverse, _ = next(iter(loader))
        torch.testing.assert_close(image[0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(reverse, torch.tensor([[0.64, 1.6, 0, 0, 0, 0]]))
    finally:
        loader.stop()


@pytest.mark.parametrize("factory", [PadAndResize, YOLOv9Augmentation])
def test_invalid_mode_rejected(factory):
    with pytest.raises(ValueError, match="resize_mode"):
        factory([64, 64], resize_mode="strech")


@pytest.mark.parametrize("nms_free", [False, True])
def test_stretch_yolov9_training_updates_weights_and_validates(tmp_path, nms_free):
    import json

    from lightning import Trainer

    from yolo.tools.solver import TrainModel
    from yolo.utils.coco_eval import CocoJsonEvaluator
    from yolo.utils.model_utils import EMA

    cfg = get_cfg(
        [
            "task=train",
            "model=v9-t",
            f"model.nms_free={str(nms_free).lower()}",
            "dataset=mock",
            "weight=false",
            "resize_mode=stretch",
            "image_size=[64,64]",
            "cpu_num=0",
            "dataset.auto_download=null",
            "dataset.class_num=1",
            f"dataset.path={tmp_path}",
            "task.data.batch_size=2",
            "task.data.equivalent_batch_size=2",
            "task.validation.data.batch_size=2",
            "task.epoch=2",
            "use_wandb=false",
        ]
    )
    cfg.task.data.data_augment.YOLOv9.albumentations = False
    images, annotations = [], []
    for index, (width, height) in enumerate([(100, 40), (40, 100)], start=1):
        for phase in ("train", "val"):
            path = tmp_path / "images" / phase / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (width, height), (index * 50, 70, 120)).save(path)
            label = tmp_path / "labels" / phase / f"{index}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.5 0.5 0.5 0.5\n")
        images.append(dict(id=index, file_name=f"{index}.png", width=width, height=height))
        annotations.append(
            dict(
                id=index,
                image_id=index,
                category_id=1,
                bbox=[width / 4, height / 4, width / 2, height / 2],
                area=width * height / 4,
                iscrowd=0,
            )
        )
    annotation = tmp_path / "annotations" / "instances_val.json"
    annotation.parent.mkdir()
    annotation.write_text(
        json.dumps(dict(info={}, images=images, annotations=annotations, categories=[dict(id=1, name="object")]))
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = TrainModel(cfg)
        before = next(model.model.parameters()).detach().clone()
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            precision="32-true",
            max_epochs=2,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            callbacks=[EMA(cfg.task.ema.decay)],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=tmp_path,
        )
        trainer.fit(model)
        assert trainer.global_step == 2
        assert isinstance(model.metric, CocoJsonEvaluator)
        assert model.post_process.nms_free == nms_free
        assert model.train_loader.dataset.transform.resize_mode == "stretch"
        assert torch.isfinite(trainer.callback_metrics["map"])
        assert not torch.equal(before, next(model.model.parameters()).detach())
        assert all(torch.isfinite(p).all() for p in model.model.parameters())
    finally:
        torch.set_num_threads(previous_threads)
