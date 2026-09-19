"""Portable runtime contracts, geometry, NMS and both actual CLI entry points."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yolo.tools.onnx_inference import ONNXDetector, class_aware_nms, run_inference


def write_model(path, *, batch=1, metadata=True, dfl=False):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    if dfl:
        predictions = np.zeros((batch, 1, 4 * 3 + 2), dtype=np.float32)
        predictions[..., [1, 4, 7, 10]] = 1  # Every distance is exactly one bin.
        predictions[..., -2:] = [0.9, 0.1]
        width, height = 32, 32
    else:
        predictions = np.array([[[16, 4, 48, 28, 0.9, 0.1]]] * batch, dtype=np.float32)
        width, height = 64, 32
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["predictions"], value=onnx.numpy_helper.from_array(predictions))],
        "portable-test",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [batch, 3, height, width])],
        [onnx.helper.make_tensor_value_info("predictions", onnx.TensorProto.FLOAT, list(predictions.shape))],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9)
    if metadata:
        onnx.helper.set_model_props(
            model,
            {
                "yolo.inference": json.dumps(
                    {
                        "version": 1,
                        "output_format": "dfl" if dfl else "xyxy",
                        "class_num": 2,
                        "class_names": ["first", "second"],
                        "reg_max": 3,
                        "strides": [32],
                    }
                )
            },
        )
    onnx.save(model, path)
    return path


def test_nms_multilabel_class_separation_and_no_second_sigmoid():
    boxes = np.array([[-20, -20, 20, 20], [-20, -20, 20, 20], [50, 50, 60, 60]], np.float32)
    scores = np.array([[0.9, 0.8], [0.7, 0.1], [0.4, 0.2]], np.float32)
    actual = class_aware_nms(boxes, scores)
    np.testing.assert_allclose(actual, [[0, -20, -20, 20, 20, 0.9], [1, -20, -20, 20, 20, 0.8]])
    assert class_aware_nms(boxes, scores, max_detections=1).shape == (1, 6)
    assert class_aware_nms(boxes, scores, confidence=1).shape == (0, 6)


def test_preprocess_rgb_normalization_and_inverse_letterbox(tmp_path):
    detector = ONNXDetector(write_model(tmp_path / "model.onnx"), threads=1)
    image = Image.new("RGB", (80, 80), (255, 0, 0))
    tensor, transform = detector.preprocess(image)
    assert tensor.shape == (3, 32, 64)
    np.testing.assert_allclose(tensor[:, 0, 0], np.full(3, 114 / 255), atol=1e-7)
    np.testing.assert_allclose(tensor[:, 10, 20], [1, 0, 0])
    np.testing.assert_allclose(detector.predict(image)[0], [[0, 0, 10, 80, 70, 0.9]], atol=1e-5)
    # Account for rounding of the resized height, rather than an idealized scale.
    _, transform = detector.preprocess(Image.new("RGB", (101, 37)))
    assert transform[:2] == (64 / 101, 23 / 37)


def test_dfl_decode_known_bins_and_metadata_precedence(tmp_path):
    detector = ONNXDetector(
        write_model(tmp_path / "model.onnx", dfl=True), class_num=80, reg_max=16, strides=[8, 16, 32], threads=1
    )
    actual = detector(np.zeros((1, 3, 32, 32), np.float32))
    np.testing.assert_allclose(actual, [[[-16, -16, 48, 48, 0.9, 0.1]]], atol=1e-6)
    assert detector.class_names == ["first", "second"]


@pytest.mark.parametrize("dfl", [False, True])
def test_old_exports_require_explicit_matching_decoder(tmp_path, dfl):
    path = write_model(tmp_path / "old.onnx", metadata=False, dfl=dfl)
    with pytest.raises(ValueError, match="no decoder metadata"):
        ONNXDetector(path, threads=1)
    detector = ONNXDetector(path, class_num=2, reg_max=3, strides=[32], threads=1)
    assert detector.output_format == ("dfl" if dfl else "xyxy")
    with pytest.raises(ValueError, match="Cannot identify"):
        ONNXDetector(path, class_num=80, threads=1)
    if dfl:
        with pytest.raises(ValueError, match="candidate count"):
            ONNXDetector(path, class_num=2, reg_max=3, threads=1)


def test_fixed_batch_final_padding_and_unsaved_run(tmp_path):
    detector = ONNXDetector(write_model(tmp_path / "batch.onnx", batch=2), threads=1)
    source = tmp_path / "inputs"
    source.mkdir()
    for index in range(3):
        Image.new("RGB", (80, 80)).save(source / f"{index}.png")
    with pytest.raises(ValueError, match="Input batch"):
        detector.predict(Image.new("RGB", (80, 80)))
    output = tmp_path / "results"
    assert run_inference(detector, source, output) == 3
    assert len(list(output.glob("frame*.jpg"))) == 3
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert [Path(row["source"]).name for row in rows] == ["0.png", "1.png", "2.png"]
    assert run_inference(detector, source) == 3


@pytest.mark.parametrize("entry", ["portable", "hydra", "hydra-no-save"])
def test_real_cli_outside_repo(tmp_path, entry):
    path = write_model(tmp_path / "model.onnx", batch=2)
    source = tmp_path / "inputs"
    source.mkdir()
    for index in range(3):
        Image.new("RGB", (80, 80)).save(source / f"{index}.png")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if entry == "portable":
        script = tmp_path / "onnx_inference.py"
        shutil.copyfile(root / "yolo/tools/onnx_inference.py", script)
        output = tmp_path / "output"
        # Isolated Python plus an import guard proves no repository/framework dependency.
        code = (
            "import runpy,sys; "
            "sys.meta_path.insert(0, type('Guard', (), {'find_spec': lambda self,name,*args: "
            "(_ for _ in ()).throw(ImportError(name)) if name.split('.')[0] in "
            "{'torch','torchvision','yolo','hydra','lightning','cv2'} else None})()); "
            f"runpy.run_path({str(script)!r}, run_name='__main__')"
        )
        command = [
            sys.executable,
            "-I",
            "-c",
            code,
            "--model",
            str(path),
            "--source",
            str(source),
            "--output",
            str(output),
            "--threads",
            "1",
        ]
    else:
        env["PYTHONPATH"] = str(root)
        output = tmp_path / "runs/inference/onnx-test"
        command = [
            sys.executable,
            "-m",
            "yolo.lazy",
            "task=inference",
            f"weight={path}",
            f"task.data.source={source}",
            "name=onnx-test",
            "cpu_num=1",
        ]
        if entry == "hydra-no-save":
            command.extend(["task.save_predict=false", "task.fast_inference=onnx"])
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = set() if entry == "hydra-no-save" else {output / f"frame{index:08d}.jpg" for index in range(3)}
    assert set(tmp_path.rglob("*.jpg")) == expected
    if expected:
        assert len((output / "predictions.jsonl").read_text().splitlines()) == 3


def test_dispatch_without_pytorch_model_or_trainer(monkeypatch):
    from tests.conftest import get_cfg
    from yolo import lazy
    from yolo.tools import onnx_runner

    cfg = get_cfg(["task=inference", "weight=example.onnx"])
    monkeypatch.setattr(onnx_runner, "run_onnx_inference", lambda actual: actual)
    monkeypatch.setattr(lazy, "setup", lambda *_: pytest.fail("ONNX must bypass Trainer/loggers"))
    assert lazy.main.__wrapped__(cfg) is cfg


def test_video_all_frames_and_read_errors(tmp_path):
    cv2 = pytest.importorskip("cv2")
    detector = ONNXDetector(write_model(tmp_path / "model.onnx"), threads=1)
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (80, 80))
    assert writer.isOpened()
    try:
        for index in range(11):
            writer.write(np.full((80, 80, 3), index * 10, dtype=np.uint8))
    finally:
        writer.release()
    assert run_inference(detector, video) == 11
    with pytest.raises(FileNotFoundError):
        run_inference(detector, tmp_path / "missing.png")
    with pytest.raises(ValueError, match="Unavailable ONNX providers"):
        ONNXDetector(tmp_path / "model.onnx", providers=["UnknownProvider"])
