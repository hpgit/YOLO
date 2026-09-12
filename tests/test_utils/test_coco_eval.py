"""Parity tests for evaluation against the original COCO annotations."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolo.utils.coco_eval import CocoJsonEvaluator


METRIC_NAMES = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)
TARGET_SIZE = (641, 639)


@pytest.fixture
def coco_fixture(tmp_path):
    image_root = tmp_path / "images"
    for relative_name in (
        "odd-frame-A.jpg",
        "nested/scene.alpha.png",
        "empty-groundtruth.jpeg",
        "dir-a/reused.jpg",
        "dir-b/reused.jpg",
    ):
        path = image_root / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    dataset = {
        "info": {"description": "synthetic COCO evaluator parity fixture"},
        "licenses": [],
        "images": [
            {"id": 101, "file_name": "odd-frame-A.jpg", "width": 1001, "height": 573},
            {"id": 205, "file_name": "nested/scene.alpha.png", "width": 537, "height": 977},
            {"id": 309, "file_name": "empty-groundtruth.jpeg", "width": 83, "height": 81},
            {"id": 401, "file_name": "dir-a/reused.jpg", "width": 51, "height": 49},
            {"id": 402, "file_name": "dir-b/reused.jpg", "width": 53, "height": 47},
        ],
        # Sorted category ids define contiguous model classes: 0 -> 3 and 1 -> 17.
        "categories": [
            {"id": 17, "name": "second"},
            {"id": 3, "name": "first"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 101,
                "category_id": 3,
                "bbox": [11, 13, 30, 20],
                # Deliberately differs from the 600-pixel bbox area and remains small.
                "area": 900,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 101,
                "category_id": 17,
                "bbox": [400, 50, 100, 100],
                # Deliberately differs from bbox area and is in COCO's medium bin.
                "area": 5000,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 101,
                "category_id": 3,
                "bbox": [700, 200, 100, 100],
                "area": 10000,
                "iscrowd": 1,
            },
            {
                "id": 4,
                "image_id": 205,
                "category_id": 3,
                "bbox": [100, 200, 300, 400],
                "area": 120000,
                "iscrowd": 0,
            },
            {
                "id": 5,
                "image_id": 205,
                "category_id": 17,
                "bbox": [5, 10, 20, 25],
                "area": 500,
                "iscrowd": 0,
            },
        ],
    }
    annotation_path = tmp_path / "instances_val.json"
    annotation_path.write_text(json.dumps(dataset), encoding="utf-8")

    native = {
        101: [
            # This high-scoring box is wholly inside top padding. COCO still
            # retains its clipped zero-area bbox, where it affects maxDets and
            # false-positive ordering.
            (1, [50, -220, 70, -210], 0.999),
            (0, [11, 13, 41, 33], 0.95),
            (1, [405, 55, 495, 145], 0.88),
            # This high-scoring detection lies within a same-category crowd box.
            (0, [710, 210, 760, 260], 0.99),
            (1, [800, 400, 850, 450], 0.40),
            # The evaluator must clip this box at the original image boundary.
            (0, [990, 560, 1020, 590], 0.20),
        ],
        205: [
            (0, [100, 200, 400, 600], 0.85),
            (1, [5, 10, 25, 35], 0.70),
            (1, [300, 700, 350, 780], 0.60),
        ],
        # A high-scoring false positive ensures that an image without GT is included.
        309: [(0, [10, 10, 40, 40], 0.97)],
    }
    return {
        "annotation_path": annotation_path,
        "image_root": image_root,
        "dataset": dataset,
        "native": native,
    }


def _model_predictions(native_rows, original_size, target_size=TARGET_SIZE):
    """Independently apply the current PadAndResize geometry to native xyxy boxes."""
    original_width, original_height = original_size
    target_width, target_height = target_size
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    gain_x = resized_width / original_width
    gain_y = resized_height / original_height

    rows = []
    for class_id, (x1, y1, x2, y2), score in native_rows:
        rows.append(
            [
                class_id,
                x1 * gain_x + pad_left,
                y1 * gain_y + pad_top,
                x2 * gain_x + pad_left,
                y2 * gain_y + pad_top,
                score,
            ]
        )
    return torch.tensor(rows, dtype=torch.float64) if rows else torch.empty((0, 6), dtype=torch.float64)


def _native_coco_detections(native_by_image):
    category_ids = (3, 17)
    image_sizes = {101: (1001, 573), 205: (537, 977), 309: (83, 81)}
    detections = []
    for image_id, native_rows in native_by_image.items():
        width, height = image_sizes[image_id]
        for class_id, (x1, y1, x2, y2), score in native_rows:
            x1 = min(max(x1, 0), width)
            x2 = min(max(x2, 0), width)
            y1 = min(max(y1, 0), height)
            y2 = min(max(y2, 0), height)
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": category_ids[class_id],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
    return detections


def _direct_coco_stats(annotation_path, detections, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(annotation_path)
        if detections:
            coco_dt = coco_gt.loadRes(detections)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "images": list(coco_gt.dataset["images"]),
                "categories": list(coco_gt.dataset["categories"]),
                "annotations": [],
            }
            coco_dt.createIndex()
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = sorted(image_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return dict(zip(METRIC_NAMES, coco_eval.stats.tolist()))


def _assert_official_metrics(actual, expected, *, atol=1e-12):
    assert set(actual) == {*METRIC_NAMES, "classes"}
    assert actual["classes"].dtype in (torch.int32, torch.int64)
    torch.testing.assert_close(actual["classes"], torch.tensor([0, 1]), rtol=0, atol=0)
    for name in METRIC_NAMES:
        assert isinstance(actual[name], torch.Tensor)
        assert actual[name].device.type == "cpu"
        assert actual[name].ndim == 0
        assert actual[name].dtype == torch.float64
        assert actual[name].item() == pytest.approx(expected[name], abs=atol, rel=0)


def _update_all(evaluator, fixture, image_ids=(101, 205, 309)):
    sizes = {image["id"]: (image["width"], image["height"]) for image in fixture["dataset"]["images"]}
    names = {image["id"]: image["file_name"] for image in fixture["dataset"]["images"]}
    evaluator.update(
        [_model_predictions(fixture["native"][image_id], sizes[image_id]) for image_id in image_ids],
        [fixture["image_root"] / names[image_id] for image_id in image_ids],
        TARGET_SIZE,
    )


def test_matches_direct_coco_stats_and_reverses_letterbox_coordinates(coco_fixture):
    evaluator = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    _update_all(evaluator, coco_fixture)

    expected = _direct_coco_stats(
        coco_fixture["annotation_path"],
        _native_coco_detections(coco_fixture["native"]),
        image_ids=(101, 205, 309),
    )
    _assert_official_metrics(evaluator.compute(), expected)

    # Check the actual serialized COCO coordinates, category mapping and boundary clipping.
    clipped = next(item for item in evaluator.predictions if item["score"] == pytest.approx(0.20))
    assert clipped["image_id"] == 101
    assert clipped["category_id"] == 3
    assert clipped["bbox"] == pytest.approx([990, 560, 11, 13], abs=1e-7)
    second_category = next(item for item in evaluator.predictions if item["score"] == pytest.approx(0.88))
    assert second_category["category_id"] == 17
    assert second_category["bbox"] == pytest.approx([405, 55, 90, 90], abs=1e-7)
    zero_area = next(item for item in evaluator.predictions if item["score"] == pytest.approx(0.999))
    assert zero_area["category_id"] == 17
    assert zero_area["bbox"] == pytest.approx([50, 0, 20, 0], abs=1e-7)


def test_split_and_reordered_batches_are_invariant(coco_fixture):
    together = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    _update_all(together, coco_fixture)

    split = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    _update_all(split, coco_fixture, image_ids=(309,))
    _update_all(split, coco_fixture, image_ids=(205, 101))

    expected = together.compute()
    actual = split.compute()
    assert [item["image_id"] for item in split.predictions] != [item["image_id"] for item in together.predictions]
    for name in (*METRIC_NAMES, "classes"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_empty_detections_and_image_without_gt_are_included(coco_fixture):
    evaluator = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    image_root = coco_fixture["image_root"]
    native = coco_fixture["native"]
    evaluator.update(
        [
            _model_predictions([native[101][1]], (1001, 573)),
            torch.empty((0, 6), dtype=torch.float64),
            _model_predictions(native[309], (83, 81)),
        ],
        [image_root / "odd-frame-A.jpg", image_root / "nested/scene.alpha.png", image_root / "empty-groundtruth.jpeg"],
        TARGET_SIZE,
    )
    selected_native = {101: [native[101][1]], 205: [], 309: native[309]}
    expected = _direct_coco_stats(
        coco_fixture["annotation_path"], _native_coco_detections(selected_native), image_ids=(101, 205, 309)
    )
    _assert_official_metrics(evaluator.compute(), expected)

    only_positive = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    only_positive.update(
        [_model_predictions([native[101][1]], (1001, 573))], [image_root / "odd-frame-A.jpg"], TARGET_SIZE
    )
    assert evaluator.compute()["map"].item() < only_positive.compute()["map"].item()


def test_reset_clears_accumulated_images_and_predictions(coco_fixture):
    reused = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    _update_all(reused, coco_fixture, image_ids=(101, 309))
    assert reused.predictions
    reused.reset()
    assert reused.predictions == []
    _update_all(reused, coco_fixture, image_ids=(205,))

    fresh = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    _update_all(fresh, coco_fixture, image_ids=(205,))
    for name in (*METRIC_NAMES, "classes"):
        torch.testing.assert_close(reused.compute()[name], fresh.compute()[name], rtol=0, atol=0)


@pytest.mark.parametrize("invalid_class", [-1.0, 2.0, 0.5])
def test_rejects_invalid_contiguous_class_ids(coco_fixture, invalid_class):
    evaluator = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    row = torch.tensor([[invalid_class, 1.0, 1.0, 2.0, 2.0, 0.5]])
    with pytest.raises(ValueError, match="class"):
        evaluator.update([row], [coco_fixture["image_root"] / "odd-frame-A.jpg"], TARGET_SIZE)


def test_rejects_unknown_and_ambiguous_filenames(coco_fixture):
    evaluator = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    empty = torch.empty((0, 6))
    with pytest.raises(ValueError, match="[Uu]nknown"):
        evaluator.update([empty], [coco_fixture["image_root"] / "missing.jpg"], TARGET_SIZE)
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        evaluator.update([empty], [Path("somewhere-else/reused.jpg")], TARGET_SIZE)

    # An exact annotation-relative path must still resolve when its basename is duplicated.
    evaluator.update([empty], [Path("dir-a/reused.jpg")], TARGET_SIZE)


def test_cwd_relative_dataset_path_wins_before_ambiguous_basename(coco_fixture):
    evaluator = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    absolute_path = coco_fixture["image_root"] / "dir-a/reused.jpg"
    cwd_relative_path = Path(os.path.relpath(absolute_path, Path.cwd()))
    evaluator.update([torch.empty((0, 6))], [cwd_relative_path], TARGET_SIZE)
    assert evaluator.image_ids == [401]


def test_duplicate_local_updates_are_idempotent_but_conflicts_fail(coco_fixture):
    evaluator = CocoJsonEvaluator(coco_fixture["annotation_path"], coco_fixture["image_root"])
    path = coco_fixture["image_root"] / "odd-frame-A.jpg"
    prediction = _model_predictions([coco_fixture["native"][101][1]], (1001, 573))
    evaluator.update([prediction], [path], TARGET_SIZE)
    evaluator.update([prediction.clone()], [path], TARGET_SIZE)
    assert len(evaluator.predictions) == 1

    conflicting = prediction.clone()
    conflicting[0, 5] = 0.123
    with pytest.raises(ValueError, match="[Cc]onflict"):
        evaluator.update([conflicting], [path], TARGET_SIZE)


def _distributed_worker(rank, world_size, init_path, annotation_path, image_root, result_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=world_size)
    try:
        evaluator = CocoJsonEvaluator(annotation_path, image_root)
        if rank == 0:
            image_ids = (101,)
            native = {101: [(0, [11, 13, 41, 33], 0.95)]}
        else:
            # The conflicting image 101 payload must lose to rank 0; image 205 is unique.
            image_ids = (101, 205)
            native = {
                101: [(1, [800, 400, 850, 450], 0.99)],
                205: [(0, [100, 200, 400, 600], 0.85)],
            }
        sizes = {101: (1001, 573), 205: (537, 977)}
        names = {101: "odd-frame-A.jpg", 205: "nested/scene.alpha.png"}
        evaluator.update(
            [_model_predictions(native[image_id], sizes[image_id]) for image_id in image_ids],
            [Path(image_root) / names[image_id] for image_id in image_ids],
            TARGET_SIZE,
        )
        torch.save(evaluator.compute(), Path(result_dir) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_distributed_compute_deduplicates_by_lowest_rank_and_broadcasts(coco_fixture, tmp_path):
    pytest.importorskip("torch.distributed")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo process group is unavailable")

    init_path = tmp_path / "gloo-init"
    result_dir = tmp_path / "distributed-results"
    result_dir.mkdir()
    mp.spawn(
        _distributed_worker,
        args=(
            2,
            str(init_path),
            str(coco_fixture["annotation_path"]),
            str(coco_fixture["image_root"]),
            str(result_dir),
        ),
        nprocs=2,
        join=True,
    )

    rank_results = [torch.load(result_dir / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    selected_native = {
        101: [(0, [11, 13, 41, 33], 0.95)],
        205: [(0, [100, 200, 400, 600], 0.85)],
    }
    expected = _direct_coco_stats(
        coco_fixture["annotation_path"], _native_coco_detections(selected_native), image_ids=(101, 205)
    )
    for result in rank_results:
        _assert_official_metrics(result, expected)
    for name in (*METRIC_NAMES, "classes"):
        torch.testing.assert_close(rank_results[0][name], rank_results[1][name], rtol=0, atol=0)
