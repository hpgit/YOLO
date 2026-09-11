"""Check the export contract against inference and real portable runtimes."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from tests.conftest import get_cfg
from yolo.model.module import Anchor2Vec
from yolo.model.yolo import create_model
from yolo.tools.export import ExportAnchor2Vec, ExportModel, export_model, validate_onnx_tensor_ranks
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
@pytest.mark.parametrize("version", ["v9-t", "v7"])
def test_onnx_runtime(tmp_path, monkeypatch, dynamic, version):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    cfg = export_cfg(
        f"model={version}", f"task.dynamic_batch={str(dynamic).lower()}", f"task.output={tmp_path / 'model.onnx'}"
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
        expected = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name)(images).numpy()
    actual = session.run(None, {"images": images.numpy()})[0]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


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
