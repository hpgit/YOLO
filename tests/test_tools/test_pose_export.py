"""Portable pose contracts preserve distributions, grid offsets, and rank limits."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from tests.conftest import get_cfg
from yolo.model.yolo import create_model
from yolo.tools.export import ExportModel, export_model, validate_onnx_tensor_ranks
from yolo.utils.bounding_box_utils import Vec2Box


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def pose_config(*overrides):
    return get_cfg(
        [
            "task=export",
            "model=v9-t-pose",
            "weight=false",
            "image_size=[64,32]",
            "dataset.class_num=1",
            "model.pose.num_keypoints=3",
            "model.pose.pose_bins=8",
            "model.pose.pose_range=4",
            *overrides,
        ]
    )


def independent_reference(model, cfg, images, probabilities):
    converter = Vec2Box(model, cfg.model.anchor, list(cfg.image_size), "cpu")
    raw = model(images, shortcut="Main")["Main"]
    pose = cfg.model.pose
    bins = torch.linspace(-pose.pose_range, pose.pose_range, pose.pose_bins)
    outputs = []
    offset = 0
    for classes, anchors, distances, keypoint_logits, visibility in raw:
        batch, _, height, width = classes.shape
        count = height * width
        classes = classes.sigmoid().flatten(2).transpose(1, 2)
        # Independent per-keypoint normalization before joining channels.
        keypoint_logits = keypoint_logits.reshape(batch, pose.num_keypoints, 2, pose.pose_bins, count)
        distributions = keypoint_logits.softmax(3).permute(0, 4, 1, 2, 3)
        visibility = visibility.sigmoid().flatten(2).transpose(1, 2)
        if probabilities:
            box_probs = anchors.softmax(1).permute(0, 3, 4, 2, 1).reshape(batch, count, -1)
            outputs.append(torch.cat((box_probs, classes, distributions.flatten(2), visibility), dim=-1))
        else:
            grid = converter.anchor_grid[offset : offset + count]
            stride = converter.scaler[offset : offset + count]
            left_top, right_bottom = (distances.flatten(2).transpose(1, 2) * stride[None, :, None]).chunk(2, -1)
            boxes = torch.cat((grid - left_top, grid + right_bottom), dim=-1)
            coordinates = (distributions * bins).sum(-1) * stride[None, :, None, None] + grid[None, :, None, :]
            keypoints = torch.cat((coordinates, visibility.unsqueeze(-1)), -1).flatten(2)
            outputs.append(torch.cat((boxes, classes, keypoints), dim=-1))
        offset += count
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("probabilities", [False, True])
def test_pose_export_matches_independent_head_reference(probabilities):
    cfg = pose_config()
    model = create_model(deepcopy(cfg.model), class_num=1, weight_path=False).eval()
    wrapper = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name, probabilities=probabilities)
    images = torch.rand(2, 3, 32, 64)
    with torch.no_grad():
        expected = independent_reference(model, cfg, images, probabilities)
        actual = wrapper(images)
    assert actual.shape == (2, 42, 116 if probabilities else 14)
    torch.testing.assert_close(actual, expected)
    assert all(tensor.ndim <= 4 for tensor in wrapper.state_dict().values())


@pytest.mark.parametrize("probabilities", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_pose_onnx_runtime_and_all_internal_ranks(tmp_path, probabilities, dynamic):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    cfg = pose_config()
    model = create_model(deepcopy(cfg.model), class_num=1, weight_path=False).eval()
    wrapper = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name, probabilities=probabilities)
    path = tmp_path / "pose.onnx"
    torch.onnx.export(
        wrapper,
        torch.zeros(1, 3, 32, 64),
        str(path),
        opset_version=17,
        dynamo=False,
        input_names=["images"],
        output_names=["predictions"],
        dynamic_axes={"images": {0: "batch"}, "predictions": {0: "batch"}} if dynamic else None,
    )
    graph = validate_onnx_tensor_ranks(onnx.load(str(path)))
    onnx.checker.check_model(graph)
    assert not any(node.op_type == "NonMaxSuppression" for node in graph.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    images = torch.rand(2 if dynamic else 1, 3, 32, 64)
    with torch.no_grad():
        expected = independent_reference(model, cfg, images, probabilities).numpy()
    actual = session.run(None, {"images": images.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_export_entrypoint_keeps_pose_channels(tmp_path):
    onnx = pytest.importorskip("onnx")
    cfg = pose_config(f"task.output={tmp_path / 'pose.onnx'}")
    path = export_model(cfg)
    graph = onnx.load(str(path))
    assert graph.graph.output[0].type.tensor_type.shape.dim[-1].dim_value == 116


def test_pose_tflite_runtime_and_all_internal_ranks(tmp_path, monkeypatch):
    pytest.importorskip("litert_torch")
    interpreter_module = pytest.importorskip("ai_edge_litert.interpreter")
    cfg = pose_config("task.format=tflite", "task.batch_size=2", f"task.output={tmp_path / 'pose.tflite'}")
    model = create_model(deepcopy(cfg.model), class_num=1, weight_path=False).eval()
    monkeypatch.setattr("yolo.tools.export.create_model", lambda *args, **kwargs: model)
    path = export_model(cfg)
    interpreter = interpreter_module.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    assert all(len(tensor["shape"]) <= 4 for tensor in interpreter.get_tensor_details())
    images = torch.rand(2, 3, 32, 64)
    with torch.no_grad():
        expected = independent_reference(model, cfg, images, probabilities=False).numpy()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], images.numpy())
    interpreter.invoke()
    actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    assert actual.shape == (2, 42, 14)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
