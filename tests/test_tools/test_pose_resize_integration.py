"""Exercise pose targets, inference inversion and COCO export across resize policies."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from yolo.tools.pose_dataset import CocoPoseDataset
from yolo.utils.pose_eval import CocoPoseEvaluator
from yolo.utils.pose_utils import reverse_pose_coordinates


@pytest.mark.parametrize("mode", ["letterbox", "stretch"])
def test_pose_dataset_inversion_and_coco_update(tmp_path, mode):
    image_dir = tmp_path / "images" / "val2017"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "sample.png"
    Image.new("RGB", (101, 57), (240, 180, 60)).save(image_path)
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    joints = [20.0, 30.0, 2.0] * 17
    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [10, 15, 50, 30],
        "area": 1500,
        "iscrowd": 0,
        "num_keypoints": 17,
        "keypoints": joints,
    }
    path = annotation_dir / "person_keypoints_val2017.json"
    path.write_text(
        json.dumps(
            {
                "info": {},
                "images": [{"id": 1, "file_name": "sample.png", "width": 101, "height": 57}],
                "categories": [{"id": 1, "name": "person", "keypoints": [str(i) for i in range(17)]}],
                "annotations": [annotation],
            }
        )
    )
    data = SimpleNamespace(image_size=[79, 81], resize_mode=mode, data_augment={})
    dataset = CocoPoseDataset(data, SimpleNamespace(path=tmp_path), "validation")
    image, targets, reverse, _ = dataset[0]
    if mode == "letterbox":
        assert image[:, 0, 0].count_nonzero() == 0
    else:
        torch.testing.assert_close(reverse, torch.tensor([79 / 101, 81 / 57, 0, 0]))
    boxes = targets[:, 1:5].unsqueeze(0)
    points = targets[:, 5:].reshape(1, 1, 17, 3)
    restored_boxes, restored_points = reverse_pose_coordinates(boxes, points, reverse.unsqueeze(0))
    torch.testing.assert_close(restored_boxes[0, 0], torch.tensor([10.0, 15.0, 60.0, 45.0]))
    torch.testing.assert_close(restored_points[0, 0], torch.tensor(joints).reshape(17, 3))
    # Main inference uses six-value stretch metadata; it must agree as well.
    metadata = torch.cat((reverse, reverse[2:])).unsqueeze(0)
    boxes6, points6 = reverse_pose_coordinates(boxes, points, metadata)
    torch.testing.assert_close(boxes6, restored_boxes)
    torch.testing.assert_close(points6, restored_points)
    prediction = torch.cat((targets[:, :5], torch.ones(1, 1), targets[:, 5:]), dim=1)
    evaluator = CocoPoseEvaluator(path)
    evaluator.update([prediction], [image_path], data.image_size, resize_mode=mode)
    detection = evaluator._records[1][0]
    assert detection["bbox"] == pytest.approx(annotation["bbox"], abs=1e-5)
    assert detection["keypoints"] == pytest.approx(joints, abs=1e-5)
    assert evaluator.compute()["map"] == pytest.approx(1.0)

    # Run the solver's actual inference route with deterministic decoded model
    # output, retaining real PostProcess, NMS and skeleton drawing downstream.
    from yolo.tools.solver import InferenceModel
    from yolo.utils.model_utils import PostProcess

    class SyntheticPoseModel:
        pose_config = {"num_keypoints": 17}

        def __call__(self, images, shortcut):
            return {"Main": None}

    class DecodedPose:
        pose_config = {"num_keypoints": 17}

        def update(self, size):
            assert size == data.image_size

        def __call__(self, prediction):
            return torch.full((1, 1, 1), 10.0), None, boxes, None, None, points

    inference = SimpleNamespace(
        model=SyntheticPoseModel(),
        cfg=SimpleNamespace(
            task=SimpleNamespace(data=data, save_json=False, save_predict=False),
            dataset=SimpleNamespace(class_list=["person"]),
        ),
        post_process=PostProcess(DecodedPose(), SimpleNamespace(min_confidence=0.1, min_iou=0.5, max_bbox=10)),
        predict_loader=SimpleNamespace(is_stream=False),
    )
    # Deliberately supply a legacy nominal letterbox transform. The solver must
    # reconstruct exact geometry from source dimensions and active mode.
    nominal_gain = min(79 / 101, 81 / 57)
    legacy = torch.tensor([[nominal_gain, 0, 18, 0, 18]])
    rendered, fps = InferenceModel.predict_step(inference, (image.unsqueeze(0), legacy, Image.open(image_path)), 0)
    assert rendered.size == (101, 57)
    assert fps is None
    result = inference.last_predictions[0]
    assert result.shape == (1, 57)
    torch.testing.assert_close(result[0, 1:5], restored_boxes[0, 0])
    torch.testing.assert_close(result[0, 6:].reshape(17, 3), restored_points[0, 0])


def test_pose_task_resize_mode_follows_global_override():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[2] / "yolo" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=["task=train-pose", "resize_mode=stretch"])
    assert cfg.task.data.resize_mode == "stretch"
    assert cfg.task.validation.data.resize_mode == "stretch"
