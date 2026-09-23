"""Check the export contract against inference and real portable runtimes."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from tests.conftest import get_cfg
from yolo.model.module import Anchor2Vec
from yolo.model.yolo import create_model
from yolo.tools.export import (
    ExportAnchor2Vec,
    ExportAnchorProbabilities,
    ExportModel,
    export_model,
    validate_onnx_tensor_ranks,
)
from yolo.utils.bounding_box_utils import create_converter


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def export_cfg(*overrides):
    return get_cfg(
        ["task=export", "model=v9-t", "weight=false", "image_size=[64,32]", "dataset.class_num=3", *overrides]
    )


@pytest.mark.parametrize("batch, reg_max", [(1, 16), (2, 8)])
def test_rank4_dfl_matches_checkpoint_projection(batch, reg_max):
    original = Anchor2Vec(reg_max).eval()
    with torch.no_grad():
        original.anc2vec.weight.copy_(torch.randn_like(original.anc2vec.weight))
    exported = ExportAnchor2Vec(original).eval()
    images = torch.randn(batch, 4 * reg_max, 3, 5)
    with torch.no_grad():
        logits, vector = exported(images)
        _, expected = original(images)
    assert logits.ndim == vector.ndim == 4
    assert all(tensor.ndim <= 4 for tensor in exported.state_dict().values())
    torch.testing.assert_close(vector, expected)


def probability_reference(predictions):
    """Use the unmodified head's rank-5 logits as an independent reference."""
    outputs = []
    for classes, logits, _ in predictions:
        # Original layout: [B, R, LTRB, H, W]. Flatten LTRB then bins.
        probabilities = logits.softmax(dim=1).permute(0, 3, 4, 2, 1)
        outputs.append(
            torch.cat((probabilities.flatten(1, 2).flatten(2), classes.sigmoid().flatten(2).transpose(1, 2)), -1)
        )
    return torch.cat(outputs, 1)


@pytest.mark.parametrize("batch, reg_max", [(1, 16), (2, 8)])
def test_rank4_box_probabilities_preserve_bin_and_direction_order(batch, reg_max):
    original = Anchor2Vec(reg_max).eval()
    exported = ExportAnchorProbabilities(original).eval()
    inputs = torch.randn(batch, 4 * reg_max, 3, 5) * 4
    with torch.no_grad():
        logits, actual = exported(inputs)
        original_logits, _ = original(inputs)
        expected = original_logits.softmax(1).permute(0, 3, 4, 2, 1).reshape(batch, 15, 4 * reg_max)
    assert logits.ndim <= 4
    assert actual.shape == (batch, 15, 4 * reg_max)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.reshape(batch, 15, 4, reg_max).sum(-1), torch.ones(batch, 15, 4))


@pytest.mark.parametrize("reg_max", [8, 16])
def test_probability_output_matches_original_head(reg_max):
    cfg = export_cfg(f"model.anchor.reg_max={reg_max}")
    model = create_model(deepcopy(cfg.model), class_num=3, weight_path=False).eval()
    wrapper = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name, probabilities=True).eval()
    assert wrapper.model is not model
    assert any(isinstance(module, Anchor2Vec) for module in model.modules())
    assert not any(isinstance(module, Anchor2Vec) for module in wrapper.modules())
    assert all(tensor.ndim <= 4 for tensor in wrapper.state_dict().values())
    images = torch.rand(2, 3, 32, 64)
    with torch.no_grad():
        expected = probability_reference(model(images, shortcut="Main")["Main"])
        actual = wrapper(images)
    assert actual.shape == (2, 42, 3 + 4 * reg_max)
    assert torch.all((actual >= 0) & (actual <= 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("version", ["v9-t", "v7"])
def test_pre_nms_matches_inference(version):
    cfg = export_cfg(f"model={version}")
    model = create_model(deepcopy(cfg.model), class_num=3, weight_path=False).eval()
    wrapper = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name).eval()
    assert wrapper.model is not model
    assert all(tensor.ndim <= 4 for tensor in wrapper.state_dict().values())
    if version == "v9-t":
        assert any(isinstance(module, Anchor2Vec) for module in model.modules())
        assert not any(isinstance(module, Anchor2Vec) for module in wrapper.modules())
    converter = create_converter(cfg.model.name, model, cfg.model.anchor, list(cfg.image_size), "cpu")
    with torch.no_grad():
        images = torch.rand(2, 3, 32, 64)
        converted = converter(model(images, shortcut="Main")["Main"])
        scores = converted[0].sigmoid()
        if len(converted) == 4:
            scores = scores * converted[3]
        expected = torch.cat((converted[2], scores), dim=-1)
        actual = wrapper(images)
    assert actual.shape == (2, 126 if version == "v7" else 42, 7)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "override, message",
    [
        ("task.format=unknown", "task.format"),
        ("task.batch_size=0", "batch_size"),
        ("image_size=[63,32]", "image_size"),
        ("task.opset=12", "opset"),
    ],
)
def test_invalid_export_options(override, message):
    with pytest.raises(ValueError, match=message):
        export_model(export_cfg(override))


def test_tflite_rejects_dynamic_batch():
    with pytest.raises(ValueError, match="fixed batch"):
        export_model(export_cfg("task.format=tflite", "task.dynamic_batch=true"))


def test_export_dispatch_bypasses_trainer(monkeypatch):
    from yolo import lazy
    from yolo.tools import export

    cfg = export_cfg()
    monkeypatch.setattr(export, "export_model", lambda actual: actual)
    monkeypatch.setattr(lazy, "setup", lambda *_: pytest.fail("Export must not initialize training/loggers"))
    assert lazy.main.__wrapped__(cfg) is cfg


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("version, reg_max", [("v9-t", 16), ("v9-t", 8), ("v7", 16)])
def test_onnx_runtime(tmp_path, monkeypatch, dynamic, version, reg_max):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    cfg = export_cfg(
        f"model={version}",
        f"task.dynamic_batch={str(dynamic).lower()}",
        f"task.output={tmp_path / 'model.onnx'}",
        *([f"model.anchor.reg_max={reg_max}"] if version == "v9-t" else []),
    )
    model = create_model(deepcopy(cfg.model), class_num=3, weight_path=False).eval()
    monkeypatch.setattr("yolo.tools.export.create_model", lambda *args, **kwargs: model)
    path = export_model(cfg)
    graph = onnx.load(str(path))
    values = [*graph.graph.input, *graph.graph.value_info, *graph.graph.output]
    assert all(v.type.tensor_type.HasField("shape") and len(v.type.tensor_type.shape.dim) <= 4 for v in values)
    assert all(len(tensor.dims) <= 4 for tensor in graph.graph.initializer)
    assert {name for node in graph.graph.node for name in node.output if name} <= {v.name for v in values}
    assert len(graph.graph.output) == 1
    assert not any("NonMaxSuppression" in node.op_type for node in graph.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    images = torch.rand(2 if dynamic else 1, 3, 32, 64)
    with torch.no_grad():
        if version == "v9-t":
            expected = probability_reference(model(images, shortcut="Main")["Main"]).numpy()
        else:
            expected = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name)(images).numpy()
    actual = session.run(None, {"images": images.numpy()})[0]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    # Exercise the portable consumer against an independent PyTorch decoder.
    from yolo.tools.onnx_inference import ONNXDetector

    detector = ONNXDetector(path, threads=2)
    with torch.no_grad():
        decoded = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name)(images).numpy()
    np.testing.assert_allclose(detector(images.numpy()), decoded, rtol=1e-4, atol=1e-4)
    if version == "v9-t":
        assert actual.shape == (images.shape[0], 42, 3 + 4 * reg_max)
        assert np.all((actual >= 0) & (actual <= 1))
        np.testing.assert_allclose(
            actual[..., : 4 * reg_max].reshape(images.shape[0], 42, 4, reg_max).sum(-1), 1, atol=1e-6
        )
        # Each head permutes L/T/R/B bins, normalizes, then flattens before concatenating.
        softmax_nodes = [node for node in graph.graph.node if node.op_type == "Softmax"]
        transpose_outputs = {
            output for node in graph.graph.node if node.op_type == "Transpose" for output in node.output
        }
        reshape_inputs = {node.input[0] for node in graph.graph.node if node.op_type == "Reshape"}
        assert len(softmax_nodes) == 12
        assert all(node.input[0] in transpose_outputs for node in softmax_nodes)
        assert all(node.output[0] in reshape_inputs for node in softmax_nodes)
        # DFL expectation and anchor/stride decoding must not leak into this graph.
        assert not any(
            "projection" in tensor.name or "anc2vec.weight" in tensor.name for tensor in graph.graph.initializer
        )


@pytest.mark.parametrize("version", ["v9-t", "v7"])
def test_tflite_runtime(tmp_path, monkeypatch, version):
    pytest.importorskip("litert_torch")
    interpreter_module = pytest.importorskip("ai_edge_litert.interpreter")
    cfg = export_cfg(
        f"model={version}", "task.format=tflite", "task.batch_size=2", f"task.output={tmp_path / 'model.tflite'}"
    )
    model = create_model(deepcopy(cfg.model), class_num=3, weight_path=False).eval()
    monkeypatch.setattr("yolo.tools.export.create_model", lambda *args, **kwargs: model)
    path = export_model(cfg)
    interpreter = interpreter_module.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    assert all(len(tensor["shape"]) <= 4 for tensor in interpreter.get_tensor_details())
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    assert len(inputs) == len(outputs) == 1
    images = torch.rand(2, 3, 32, 64)
    with torch.no_grad():
        expected = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name)(images).numpy()
    interpreter.set_tensor(inputs[0]["index"], images.numpy())
    interpreter.invoke()
    actual = interpreter.get_tensor(outputs[0]["index"])
    assert actual.shape == (2, 126 if version == "v7" else 42, 7)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_onnx_rank_guard_rejects_rank5():
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        "rank5",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2, 3, 4, 5])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2, 3, 4, 5])],
    )
    with pytest.raises(ValueError, match="rank 5"):
        validate_onnx_tensor_ranks(onnx.helper.make_model(graph))


def test_onnx_rank_guard_rejects_internal_rank5_with_flat_output():
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "Constant", [], ["internal"], value=onnx.numpy_helper.from_array(np.zeros((1, 2, 3, 4, 5), np.float32))
            ),
            onnx.helper.make_node("Flatten", ["internal"], ["y"]),
        ],
        "internal_rank5",
        [],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 120])],
    )
    with pytest.raises(ValueError, match="rank 5"):
        validate_onnx_tensor_ranks(onnx.helper.make_model(graph))
