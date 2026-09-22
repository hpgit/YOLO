"""One-to-one postprocessing and actual ONNX export/runtime integration."""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image

from tests.conftest import get_cfg
from yolo.config.config import NMSConfig
from yolo.model.yolo import create_model
from yolo.tools.export import ExportModel, export_model
from yolo.tools.onnx_inference import ONNXDetector, nms_free_topk
from yolo.utils.bounding_box_utils import create_converter
from yolo.utils.model_utils import PostProcess
from yolo.utils.nms_utils import select_nms_free_detections


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_topk_retains_overlaps_one_class_per_anchor_and_stable_ties():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [2, 2, 8, 8], [30, 30, 40, 40]], np.float32)
    scores = np.array([[0.9, 0.8], [0.9, 0.2], [0.1, 0.95], [0.5, 0.2]], np.float32)
    expected = np.array([[1, 2, 2, 8, 8, 0.95], [0, 0, 0, 10, 10, 0.9], [0, 0, 0, 10, 10, 0.9]], np.float32)
    np.testing.assert_allclose(nms_free_topk(boxes, scores), expected)
    actual = select_nms_free_detections(torch.tensor(boxes)[None], torch.tensor(scores)[None])[0]
    np.testing.assert_allclose(actual.numpy(), expected)
    np.testing.assert_allclose(nms_free_topk(boxes, scores, max_detections=2), expected[:2])
    # The second tied anchor remains first if its coordinates identify it.
    boxes[1] = [1, 1, 11, 11]
    assert nms_free_topk(boxes, scores)[1:3, 1].tolist() == [0, 1]


def test_topk_batch_caps_empty_inputs_and_nonfinite_predictions():
    boxes = torch.tensor([[[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]]] * 2, dtype=torch.float32)
    scores = torch.tensor([[[0.9, 0.1], [0.8, 0.1], [0.7, 0.1]], [[0.1, 0.2]] * 3])
    actual = select_nms_free_detections(boxes, scores, max_detections=1)
    assert [tuple(result.shape) for result in actual] == [(1, 6), (0, 6)]
    assert select_nms_free_detections(boxes[:, :0], scores[:, :0])[0].shape == (0, 6)
    assert nms_free_topk(boxes[0, :0].numpy(), scores[0, :0].numpy()).shape == (0, 6)
    boxes[0, 0, 0] = float("nan")
    boxes[0, 1, 2] = boxes[0, 1, 0]
    scores[0, 2, 0] = float("nan")
    assert select_nms_free_detections(boxes, scores)[0].shape == (0, 6)
    assert nms_free_topk(boxes[0].numpy(), scores[0].numpy()).shape == (0, 6)


def test_postprocess_never_calls_nms_and_restores_coordinates(monkeypatch):
    monkeypatch.setattr("yolo.utils.model_utils.bbox_nms", lambda *_: pytest.fail("NMS-free called NMS"))
    boxes = torch.tensor([[[4, 8, 24, 28], [4, 8, 24, 28]]], dtype=torch.float32)
    scores = torch.tensor([[[0.9, 0.8], [0.7, 0.1]]])

    class Converter:
        def __call__(self, prediction):
            assert prediction == "one-to-one"
            return torch.logit(scores), None, boxes

        def update(self, image_size):
            assert image_size == [32, 32]

    postprocess = PostProcess(Converter(), NMSConfig(0.5, 0.0, 300), nms_free=True)
    actual = postprocess({"Main": "one-to-one"}, torch.tensor([[2, 4, 8, 4, 8]]), image_size=[32, 32])[0]
    torch.testing.assert_close(actual, torch.tensor([[0, 0, 0, 10, 10, 0.9], [0, 0, 0, 10, 10, 0.7]]))


def constant_export(path, postprocess_metadata=None):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    predictions = np.array([[[4, 4, 28, 28, 0.9, 0.1], [4, 4, 28, 28, 0.7, 0.1]]], np.float32)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["predictions"], value=onnx.numpy_helper.from_array(predictions))],
        "nms-free-test",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 32, 32])],
        [onnx.helper.make_tensor_value_info("predictions", onnx.TensorProto.FLOAT, list(predictions.shape))],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9)
    metadata = {"version": 1, "output_format": "xyxy", "class_num": 2}
    metadata.update(postprocess_metadata or {})
    onnx.helper.set_model_props(model, {"yolo.inference": json.dumps(metadata)})
    onnx.save(model, path)
    return path


@pytest.mark.parametrize(
    "metadata, fallback, expected",
    [
        ({"nms_free": True, "postprocess": "topk"}, False, 2),
        ({"nms_free": False, "postprocess": "nms"}, True, 1),
        ({"postprocess": "topk"}, False, 2),
        ({}, False, 1),
        ({}, True, 2),
    ],
)
def test_runtime_metadata_routes_topk_and_preserves_legacy_nms(tmp_path, monkeypatch, metadata, fallback, expected):
    detector = ONNXDetector(constant_export(tmp_path / "constant.onnx", metadata), nms_free=fallback, threads=1)
    if expected == 2:
        monkeypatch.setattr("yolo.tools.onnx_inference.class_aware_nms", lambda *_: pytest.fail("NMS-free called NMS"))
    result = detector.predict(Image.new("RGB", (64, 64)))[0]
    assert result.shape == (expected, 6)
    np.testing.assert_allclose(result[:, 1:5], np.array([[8, 8, 56, 56]] * expected))


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"nms_free": "true"}, "boolean"),
        ({"postprocess": "unknown"}, "Unsupported ONNX postprocess"),
        ({"nms_free": True, "postprocess": "nms"}, "disagree"),
    ],
)
def test_runtime_rejects_invalid_postprocessing_metadata(tmp_path, metadata, message):
    with pytest.raises(ValueError, match=message):
        ONNXDetector(constant_export(tmp_path / "constant.onnx", metadata), threads=1)


@pytest.mark.parametrize("dynamic", [False, True])
def test_actual_nms_free_export_uses_only_o2o_rank4_and_runtime_topk(tmp_path, monkeypatch, dynamic):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    cfg = get_cfg(
        [
            "task=export",
            "model=v9-t",
            "model.nms_free=true",
            "weight=false",
            "image_size=[64,32]",
            "dataset.class_num=3",
            f"task.dynamic_batch={str(dynamic).lower()}",
            f"task.output={tmp_path / 'nms-free.onnx'}",
        ]
    )
    model = create_model(deepcopy(cfg.model), class_num=3, weight_path=False).eval()
    main = next(layer for layer in model.model if layer.tags == "Main")
    with torch.no_grad():
        for index, head in enumerate(main.one2one_heads):
            head.class_conv[-1].weight.zero_()
            head.class_conv[-1].bias.copy_(torch.tensor([1.5 - index * 0.2, -1, 0.1]))
    monkeypatch.setattr("yolo.tools.export.create_model", lambda *args, **kwargs: model)
    path = export_model(cfg)
    graph = onnx.load(path)
    metadata = json.loads(next(prop.value for prop in graph.metadata_props if prop.key == "yolo.inference"))
    assert metadata["nms_free"] is True and metadata["postprocess"] == "topk"
    assert not any(node.op_type == "NonMaxSuppression" for node in graph.graph.node)
    assert all(len(value.type.tensor_type.shape.dim) <= 4 for value in graph.graph.value_info)
    names = [tensor.name for tensor in graph.graph.initializer]
    assert any("one2one_heads" in name for name in names)
    assert not any(".heads." in name for name in names)
    detector = ONNXDetector(path, confidence=0.5, max_detections=7, nms_free=False, threads=2)
    assert detector.nms_free
    count = 2 if dynamic else 1
    images = [Image.new("RGB", (64, 32), (20 * index, 40, 80)) for index in range(count)]
    tensors = torch.tensor(np.stack([detector.preprocess(image)[0] for image in images]))
    converter = create_converter(cfg.model.name, model, cfg.model.anchor, list(cfg.image_size), "cpu")
    with torch.no_grad():
        expected_raw = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name)(tensors).numpy()
        expected = PostProcess(converter, NMSConfig(0.5, 0.0, 7), nms_free=True)(model(tensors))
    np.testing.assert_allclose(detector(tensors.numpy()), expected_raw, atol=1e-4, rtol=1e-4)
    monkeypatch.setattr("yolo.tools.onnx_inference.class_aware_nms", lambda *_: pytest.fail("NMS-free called NMS"))
    actual = detector.predict(images)
    assert len(actual) == count
    for result, reference in zip(actual, expected):
        assert result.shape == (7, 6)
        reference[:, [1, 3]] = reference[:, [1, 3]].clamp(0, 64)
        reference[:, [2, 4]] = reference[:, [2, 4]].clamp(0, 32)
        np.testing.assert_allclose(result, reference.numpy(), atol=1e-4, rtol=1e-4)
