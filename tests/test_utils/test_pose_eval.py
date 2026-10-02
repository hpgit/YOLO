"""COCO OKS protocol and coordinate inversion tests using genuine COCOeval."""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from yolo.utils.coco_eval import CocoJsonEvaluator
from yolo.utils.pose_eval import POSE_METRIC_NAMES, CocoPoseEvaluator

TARGET_SIZE = (641, 639)


@pytest.fixture
def pose_coco(tmp_path):
    keypoints = [value for index in range(17) for value in (100 + index * 3, 120 + index * 2, 1 if index % 2 else 2)]
    keypoints[-3:] = [0, 0, 0]
    dataset = {
        "info": {},
        "images": [
            {"id": 11, "file_name": "first.jpg", "width": 1001, "height": 573},
            {"id": 25, "file_name": "second.jpg", "width": 537, "height": 977},
        ],
        "categories": [{"id": 1, "name": "person", "keypoints": [str(i) for i in range(17)], "skeleton": []}],
        "annotations": [
            {
                "id": 1,
                "image_id": 11,
                "category_id": 1,
                "bbox": [80, 90, 100, 100],
                "area": 5000,
                "iscrowd": 0,
                "num_keypoints": 16,
                "keypoints": keypoints,
            },
            {
                "id": 2,
                "image_id": 25,
                "category_id": 1,
                "bbox": [50, 80, 200, 200],
                "area": 40000,
                "iscrowd": 0,
                "num_keypoints": 16,
                "keypoints": keypoints,
            },
        ],
    }
    path = tmp_path / "person_keypoints_val.json"
    path.write_text(json.dumps(dataset))
    return path, dataset


def letterboxed_prediction(dataset, index=0):
    image, annotation = dataset["images"][index], dataset["annotations"][index]
    original_width, original_height = image["width"], image["height"]
    width, height = TARGET_SIZE
    scale = min(width / original_width, height / original_height)
    resized_width, resized_height = int(original_width * scale), int(original_height * scale)
    gain_x, gain_y = resized_width / original_width, resized_height / original_height
    pad_left, pad_top = (width - resized_width) // 2, (height - resized_height) // 2
    x, y, w, h = annotation["bbox"]
    joints = torch.tensor(annotation["keypoints"], dtype=torch.float64).reshape(17, 3)
    joints[:, 0] = joints[:, 0] * gain_x + pad_left
    joints[:, 1] = joints[:, 1] * gain_y + pad_top
    joints[:, 2] = 0.8
    row = [
        0,
        x * gain_x + pad_left,
        y * gain_y + pad_top,
        (x + w) * gain_x + pad_left,
        (y + h) * gain_y + pad_top,
        0.95,
    ]
    return torch.tensor([row + joints.flatten().tolist()], dtype=torch.float64)


def test_perfect_predictions_use_official_oks_and_ar20(pose_coco):
    path, dataset = pose_coco
    evaluator = CocoPoseEvaluator(path)
    assert isinstance(evaluator, CocoJsonEvaluator)
    original_annotations = copy.deepcopy(evaluator.coco_gt.dataset["annotations"])
    evaluator.update(
        [letterboxed_prediction(dataset, 0), letterboxed_prediction(dataset, 1)],
        ["first.jpg", "second.jpg"],
        TARGET_SIZE,
    )
    result = evaluator.compute()
    assert set(result) == set(POSE_METRIC_NAMES) | {"classes"}
    for key in POSE_METRIC_NAMES:
        assert result[key].item() == pytest.approx(1.0)
    assert "mar_100" not in result
    for actual, expected in zip(evaluator.coco_gt.dataset["annotations"], original_annotations):
        for field in ("keypoints", "area", "iscrowd", "num_keypoints"):
            assert actual[field] == expected[field]


def test_odd_aspect_ratio_inverse_is_exact_and_preserves_confidence(pose_coco):
    path, dataset = pose_coco
    evaluator = CocoPoseEvaluator(path)
    for index in range(2):
        prediction = letterboxed_prediction(dataset, index)
        original = prediction.clone()
        evaluator.update([prediction], [dataset["images"][index]["file_name"]], TARGET_SIZE)
        torch.testing.assert_close(prediction, original)
        exported = evaluator.predictions[index]
        actual = torch.tensor(exported["keypoints"], dtype=torch.float64).reshape(17, 3)
        expected = torch.tensor(dataset["annotations"][index]["keypoints"], dtype=torch.float64).reshape(17, 3)
        torch.testing.assert_close(actual[:, :2], expected[:, :2], rtol=0, atol=1e-12)
        torch.testing.assert_close(actual[:, 2], torch.full((17,), 0.8, dtype=torch.float64))
        assert exported["score"] == 0.95
        assert exported["bbox"] == pytest.approx(dataset["annotations"][index]["bbox"])


def test_evaluates_only_observed_subset_and_empty_predictions(pose_coco):
    path, dataset = pose_coco
    evaluator = CocoPoseEvaluator(path)
    evaluator.update([letterboxed_prediction(dataset)], ["first.jpg"], TARGET_SIZE)
    result = evaluator.compute()
    assert evaluator.image_ids == [11]
    assert result["map"].item() == pytest.approx(1.0)
    assert result["mar_20"].item() == pytest.approx(1.0)
    assert result["map_large"].item() == -1
    evaluator.reset()
    evaluator.update([torch.empty(0, 57)], ["first.jpg"], TARGET_SIZE)
    result = evaluator.compute()
    assert result["map"].item() == 0
    assert result["mar_20"].item() == 0
    assert evaluator.predictions == [] and evaluator.image_ids == [11]


def test_keypoint_errors_affect_ap_even_when_boxes_are_perfect(pose_coco):
    path, dataset = pose_coco
    evaluator = CocoPoseEvaluator(path)
    prediction = letterboxed_prediction(dataset)
    prediction[:, 6::3] += 200
    evaluator.update([prediction], ["first.jpg"], TARGET_SIZE)
    assert evaluator.compute()["map"].item() == 0


def test_outside_image_keypoints_are_not_clipped(pose_coco):
    path, dataset = pose_coco
    evaluator = CocoPoseEvaluator(path)
    prediction = letterboxed_prediction(dataset)
    prediction[0, 6] = -100
    evaluator.update([prediction], ["first.jpg"], TARGET_SIZE)
    assert evaluator.predictions[0]["keypoints"][0] < 0


@pytest.mark.parametrize("columns", [6, 54, 56, 58])
def test_rejects_malformed_prediction_width(pose_coco, columns):
    evaluator = CocoPoseEvaluator(pose_coco[0])
    with pytest.raises(ValueError, match="shape"):
        evaluator.update([torch.empty(0, columns)], ["first.jpg"], TARGET_SIZE)


@pytest.mark.parametrize("column", [0, 1, 5, 6, 7, 8, 56])
def test_rejects_nonfinite_prediction_values(pose_coco, column):
    path, dataset = pose_coco
    evaluator = CocoPoseEvaluator(path)
    prediction = letterboxed_prediction(dataset)
    prediction[0, column] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        evaluator.update([prediction], ["first.jpg"], TARGET_SIZE)


def test_rejects_non17_layout_and_malformed_ground_truth(pose_coco):
    path, dataset = pose_coco
    with pytest.raises(ValueError, match="17-keypoint"):
        CocoPoseEvaluator(path, num_keypoints=16)
    dataset["annotations"][0]["keypoints"] = [1, 2, 2] * 16
    path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="17 keypoint"):
        CocoPoseEvaluator(path)


def test_no_observed_images_is_an_error(pose_coco):
    with pytest.raises(ValueError, match="no images"):
        CocoPoseEvaluator(pose_coco[0]).compute()


def _pose_distributed_worker(rank, init_path, annotation_path, result_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        dataset = json.loads(Path(annotation_path).read_text())
        evaluator = CocoPoseEvaluator(annotation_path)
        evaluator.update([letterboxed_prediction(dataset, rank)], [dataset["images"][rank]["file_name"]], TARGET_SIZE)
        if rank == 1:
            # Padding duplicate with conflicting payload loses to rank zero.
            evaluator.update([torch.empty(0, 57)], ["first.jpg"], TARGET_SIZE)
        torch.save(evaluator.compute(), Path(result_dir) / f"pose-rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_distributed_pose_gather_deduplicates_and_broadcasts_ar20(pose_coco, tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo process group is unavailable")
    mp.spawn(
        _pose_distributed_worker,
        args=(str(tmp_path / "pose-gloo"), str(pose_coco[0]), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    results = [torch.load(tmp_path / f"pose-rank-{rank}.pt", weights_only=True) for rank in range(2)]
    for result in results:
        assert set(result) == set(POSE_METRIC_NAMES) | {"classes"}
        for name in POSE_METRIC_NAMES:
            assert result[name].item() == pytest.approx(1.0)
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], rtol=0, atol=0)
