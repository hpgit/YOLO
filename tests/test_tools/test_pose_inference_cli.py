"""Actual pose CLI checkpoint loading and rank-zero JPEG/JSON output."""

import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from PIL import Image

from tests.conftest import get_cfg
from yolo.model.yolo import create_model


@pytest.fixture(scope="module")
def synthetic_pose_checkpoint(tmp_path_factory):
    cfg = get_cfg(["model=v9-t-pose", "dataset=coco-pose", "weight=false"])
    model = create_model(deepcopy(cfg.model), class_num=1, weight_path=False).eval()
    # Deliberately synthetic high scores exercise nonempty output and rendering;
    # this checkpoint does not represent trained detection/pose accuracy.
    with torch.no_grad():
        for layer in model.model:
            if layer.tags in ("Main", "AUX"):
                for head in layer.heads:
                    head.class_conv[-1].bias.fill_(4)
                    head.visibility_conv.bias.fill_(4)
    destination = tmp_path_factory.mktemp("synthetic-pose-checkpoint") / "pose.pt"
    torch.save({"weights": model.model.state_dict(), "pose_config": model.pose_config}, destination)
    return destination


@pytest.mark.parametrize(
    "devices,save_predict,save_json",
    [
        (1, True, True),
        (2, True, True),
        (2, False, False),
        (1, False, True),
        (1, True, False),
    ],
)
def test_pose_cli_outputs_only_under_run_directory(
    tmp_path, synthetic_pose_checkpoint, devices, save_predict, save_json
):
    source = tmp_path / "inputs"
    source.mkdir()
    for index in range(2):
        Image.new("RGB", (97, 53), (index * 80, 30, 60)).save(source / f"{index}.png")
    project_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(project_root), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "yolo.lazy",
            "task=inference",
            "model=v9-t-pose",
            "dataset=coco-pose",
            f"weight={synthetic_pose_checkpoint}",
            "accelerator=cpu",
            f"device={devices}",
            "precision=32-true",
            "image_size=[64,64]",
            f"task.data.source={source}",
            f"task.save_predict={str(save_predict).lower()}",
            f"task.save_json={str(save_json).lower()}",
            "task.nms.min_confidence=0.01",
            "task.nms.max_bbox=3",
            "task.keypoint_confidence=0.5",
            "use_wandb=false",
            "use_tensorboard=false",
            "name=pose-cli-test",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run_dir = tmp_path / "runs" / "inference" / "pose-cli-test"
    expected_images = {run_dir / f"frame{index:08d}.jpg" for index in range(2)} if save_predict else set()
    expected_json = {run_dir / f"frame{index:08d}.json" for index in range(2)} if save_json else set()
    assert set(tmp_path.rglob("*.jpg")) == expected_images
    assert set(tmp_path.rglob("*.json")) == expected_json
    for path in expected_images:
        with Image.open(path) as frame:
            assert frame.format == "JPEG"
            assert frame.size == (97, 53)
    for path in expected_json:
        payload = json.loads(path.read_text())
        assert payload["image_size"] == [97, 53]
        assert 1 <= len(payload["instances"]) <= 3
        for person in payload["instances"]:
            assert person["class_id"] == 0
            assert len(person["bbox_xyxy"]) == 4
            assert len(person["keypoints"]) == 17
            assert 0 <= person["score"] <= 1
            assert all(math.isfinite(value) for value in person["bbox_xyxy"])
            assert all(
                len(point) == 3 and all(math.isfinite(value) for value in point) and 0 <= point[2] <= 1
                for point in person["keypoints"]
            )
