"""Deploy fusion, real gradients, checkpoint lifecycle and QDQ runtime contracts."""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader

from tests.conftest import get_cfg
from yolo.model.module import FixedKernelConv2d, RepConv
from yolo.model.yolo import create_model
from yolo.tools.export import ExportModel, export_model, validate_onnx_tensor_ranks
from yolo.tools.loss_functions import create_loss_function
from yolo.tools.qat import (
    QATConv2d,
    QATFixedKernelConv2d,
    checkpoint_weights,
    configure_qat_run,
    deploy_model,
    encoding_manifest,
    freeze_for_export,
    prepare_qat,
    quantizers,
    set_qat_epoch,
)
from yolo.tools.solver import TrainModel
from yolo.utils.bounding_box_utils import create_converter
from yolo.utils.checkpoint_utils import YOLOCheckpoint


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def config(*overrides):
    return get_cfg(
        [
            "task=train",
            "model=v9-t",
            "weight=false",
            "dataset=mock",
            "dataset.class_num=3",
            "image_size=[64,32]",
            "qat.enabled=true",
            "qat.fake_quant_start_epoch=0",
            "qat.observer_freeze_epoch=1",
            *overrides,
        ]
    )


def model_for(cfg):
    return create_model(deepcopy(cfg.model), False, cfg.dataset.class_num, qat_cfg=cfg.qat)


def calibrated(cfg):
    model = model_for(cfg).train()
    for _ in range(2):
        model(torch.rand(2, 3, 32, 64))
    return model.eval()


def test_deploy_fusion_preserves_float_main_outputs():
    cfg = config()
    source = create_model(deepcopy(cfg.model), False, 3).eval()
    with torch.no_grad():
        for layer in source.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.running_mean.uniform_(-0.1, 0.1)
                layer.running_var.uniform_(0.5, 1.5)
        inputs = torch.rand(2, 3, 32, 64)
        expected = source(inputs, shortcut="Main")["Main"]
        deployed = deploy_model(deepcopy(source))
        actual = deployed(inputs)
    assert list(actual) == ["Main"]
    assert not any(isinstance(layer, nn.BatchNorm2d) for layer in deployed.modules())
    assert all(
        hasattr(layer, "reparam") and not hasattr(layer, "conv1")
        for layer in deployed.modules()
        if isinstance(layer, RepConv)
    )
    for left, right in zip(expected, actual["Main"]):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("model_name", ["v9-t", "v9-sr-t", "v9-sr-c"])
def test_real_detection_loss_backpropagates_and_observers_freeze(model_name):
    cfg = config(f"model={model_name}")
    model = model_for(cfg).train()
    converter = create_converter(cfg.model.name, model, cfg.model.anchor, list(cfg.image_size), "cpu")
    loss_fn = create_loss_function(cfg, converter)
    conv = next(layer for layer in model.modules() if isinstance(layer, QATConv2d))
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    fixed = [(layer, layer.weight.clone()) for layer in model.modules() if isinstance(layer, FixedKernelConv2d)]
    images = torch.rand(2, 3, 32, 64)
    targets = torch.tensor([[[1, 8, 4, 48, 28]], [[2, 12, 5, 52, 29]]], dtype=torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss, items = loss_fn(None, converter(model(images)["Main"]), targets)
    assert torch.isfinite(loss) and all(torch.isfinite(item) for item in items.values())
    loss.backward()
    assert conv.weight.grad is not None and conv.weight.grad.abs().sum() > 0
    optimizer.step()
    for layer, weight in fixed:
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
        assert layer.weight.grad is None and not layer.weight.requires_grad
    assert any(not torch.equal(before, after) for before, after in zip(initial, model.parameters()))
    state = deepcopy(encoding_manifest(model))
    stats = {name: q.activation_post_process.min_val.clone() for name, q in quantizers(model)}
    model.eval()(images * 100)
    assert encoding_manifest(model) == state
    for name, q in quantizers(model):
        torch.testing.assert_close(stats[name], q.activation_post_process.min_val)
    set_qat_epoch(model, 1)
    model.train()(images * 100)
    assert encoding_manifest(model) == state
    assert all(q.fake_quant_enabled.item() == 1 and q.observer_enabled.item() == 0 for _, q in quantizers(model))


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("model_name", ["v9-t", "v9-sr-t", "v9-sr-c"])
def test_qdq_runtime_encodings_and_rank(tmp_path, dynamic, model_name):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    cfg = config(f"model={model_name}")
    model = calibrated(cfg)
    path = tmp_path / "qat.pt"
    torch.save(checkpoint_weights(model), path)
    export_cfg = get_cfg(
        [
            "task=export",
            f"model={model_name}",
            "dataset.class_num=3",
            "image_size=[64,32]",
            f"weight={path}",
            "task.qdq=true",
            f"task.dynamic_batch={str(dynamic).lower()}",
            f'task.output={tmp_path / "qat.onnx"}',
        ]
    )
    exported_path = export_model(export_cfg)
    restored = create_model(deepcopy(cfg.model), path, 3).eval()
    assert encoding_manifest(restored) == encoding_manifest(model)
    graph = validate_onnx_tensor_ranks(onnx.load(exported_path))
    onnx.checker.check_model(graph)
    metadata = {item.key: item.value for item in graph.metadata_props}
    manifest = json.loads(metadata["yolo.qat.encodings"])
    q_nodes = [node for node in graph.graph.node if node.op_type == "QuantizeLinear"]
    smoothing_count = sum(isinstance(layer, QATFixedKernelConv2d) for layer in model.modules())
    assert (
        len(q_nodes)
        == len(manifest)
        == (3 * sum(isinstance(layer, QATConv2d) for layer in model.modules()) + smoothing_count)
    )
    # Each fixed smoothing Conv must consume its own input Q/DQ directly.
    producers = {output: node for node in graph.graph.node for output in node.output}
    smoothing_nodes = [node for node in graph.graph.node if node.op_type == "Conv" and "/avg_pool/" in node.name]
    assert len(smoothing_nodes) == smoothing_count
    for node in smoothing_nodes:
        dq = producers[node.input[0]]
        assert dq.op_type == "DequantizeLinear"
        q = producers[dq.input[0]]
        assert q.op_type == "QuantizeLinear" and "/avg_pool/input_fake_quant/" in q.name
        assert producers[node.input[1]].op_type == "Constant"
    constants = {tensor.name: onnx.numpy_helper.to_array(tensor) for tensor in graph.graph.initializer}
    for node in graph.graph.node:
        if node.op_type == "Identity" and node.input[0] in constants:
            constants[node.output[0]] = constants[node.input[0]]
        if node.op_type == "Constant":
            constants[node.output[0]] = onnx.numpy_helper.to_array(node.attribute[0].t)
    # Check every exported encoding, not only Q/DQ node presence.
    original_modules = dict(
        ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name, probabilities=True).named_modules()
    )
    for node in q_nodes:
        name = node.input[1].removesuffix(".scale")
        entry = manifest[name]
        np.testing.assert_array_equal(constants[node.input[1]], entry["scale"])
        np.testing.assert_array_equal(constants[node.input[2]], entry["zero_point"])
        assert constants[node.input[2]].dtype == np.dtype(entry["dtype"])
        axes = [attr.i for attr in node.attribute if attr.name == "axis"]
        assert axes == ([] if entry["axis"] is None else [0])
        if entry["axis"] == 0:
            # Float initializers have been snapped to the trained integer grid;
            # Q/DQ must reproduce the original native fake-quantized weights.
            conv = original_modules[name.removesuffix(".weight_fake_quant")]
            reference = conv.weight_fake_quant(conv.weight).detach().numpy()
            scale = constants[node.input[1]].reshape(-1, 1, 1, 1)
            actual_weight = np.clip(np.rint(constants[node.input[0]] / scale), -128, 127) * scale
            np.testing.assert_array_equal(actual_weight, reference)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    # Compare Q/DQ semantics before backend-specific integer kernel rewrites.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(exported_path), sess_options=options, providers=["CPUExecutionProvider"])
    images = torch.rand(2 if dynamic else 1, 3, 32, 64)
    freeze_for_export(model)
    with torch.no_grad():
        expected = ExportModel(model, cfg.model.anchor, list(cfg.image_size), cfg.model.name, probabilities=True)(
            images
        ).numpy()
    actual = session.run(None, {"images": images.numpy()})[0]
    assert actual.shape == (images.shape[0], 42, 67)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(actual[..., :64].reshape(-1, 4, 16).sum(-1), 1, atol=1e-6)
    # Smoke-test the deploy consumer with runtime optimizations enabled, too.
    from yolo.tools.onnx_inference import ONNXDetector

    decoded = ONNXDetector(exported_path, threads=2)(images.numpy())
    assert decoded.shape == (images.shape[0], 42, 7)
    assert np.isfinite(decoded).all()
    assert np.all((decoded[..., 4:] >= 0) & (decoded[..., 4:] <= 1))


def test_fail_closed_and_profile_restore(tmp_path):
    cfg = config()
    model = model_for(cfg)
    with pytest.raises(ValueError, match="Uninitialized"):
        freeze_for_export(model)
    model.train()(torch.rand(2, 3, 32, 64))
    for _, q in quantizers(model):
        q.disable_fake_quant()
    with pytest.raises(ValueError, match="warmup"):
        freeze_for_export(model)
    path = tmp_path / "qat.pt"
    torch.save(checkpoint_weights(model), path)
    with pytest.raises(ValueError, match="class_num mismatch"):
        create_model(deepcopy(cfg.model), path, 4)
    cfg.weight = str(path)
    cfg.qat.enabled = False
    configure_qat_run(cfg)
    assert cfg.qat.enabled
    cfg.qat.averaging_constant = 0.2
    with pytest.raises(ValueError, match="differs"):
        configure_qat_run(cfg)
    fp = create_model(deepcopy(cfg.model), False, 3)
    with pytest.raises(ValueError, match="trained QAT"):
        freeze_for_export(fp)
    wrong = config("model=v7")
    with pytest.raises(ValueError, match="YOLOv9"):
        model_for(wrong)


@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize("model_name", ["v9-t", "v9-sr-t"])
def test_lightning_train_resume_and_best_checkpoint(tmp_path, monkeypatch, warmup, model_name):
    first_epochs = 2 if warmup else 1
    cfg = config(
        f"model={model_name}",
        f"task.epoch={first_epochs + 1}",
        f"qat.fake_quant_start_epoch={int(warmup)}",
        f"qat.observer_freeze_epoch={first_epochs}",
        "task.data.batch_size=2",
        "task.data.equivalent_batch_size=2",
        "task.optimizer.args.lr=0.0001",
        "task.validation.evaluator=torchmetrics",
    )
    images = torch.rand(2, 3, 32, 64)
    targets = torch.tensor([[[1, 8, 4, 48, 28]], [[2, 12, 5, 52, 29]]], dtype=torch.float32)

    def collate(_):
        return 2, images, targets, None, ["first.jpg", "second.jpg"]

    loader = DataLoader(range(4), batch_size=2, collate_fn=collate)
    monkeypatch.setattr("yolo.tools.solver.create_dataloader", lambda *args, **kwargs: loader)
    callback = YOLOCheckpoint(tmp_path)

    def trainer(epochs, callbacks):
        return Trainer(
            accelerator="cpu",
            devices=1,
            precision="32-true",
            max_epochs=epochs,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=1,
            default_root_dir=tmp_path,
        )

    module = TrainModel(deepcopy(cfg))
    run = trainer(first_epochs, [callback])
    run.fit(module)
    checkpoint = callback.last_model_path or callback.best_model_path
    assert checkpoint and (tmp_path / "best.pt").exists()
    best = create_model(deepcopy(cfg.model), tmp_path / "best.pt", 3)
    freeze_for_export(best)
    assert encoding_manifest(best) == encoding_manifest(module.model)
    resumed = TrainModel(deepcopy(cfg))
    second = trainer(first_epochs + 1, [])
    second.fit(resumed, ckpt_path=checkpoint)
    assert second.global_step == 2 * (first_epochs + 1)
    assert all(q.observer_enabled.item() == 0 for _, q in quantizers(resumed.model))
    # Frozen observers survive both loading and another epoch of real optimization.
    assert encoding_manifest(resumed.model) == encoding_manifest(module.model)
    assert any(not torch.equal(a, b) for a, b in zip(resumed.model.parameters(), module.model.parameters()))


@pytest.mark.parametrize("model_name", ["v9-t", "v9-sr-t", "v9-sr-s", "v9-sr-m", "v9-sr-c"])
def test_warmup_schedule_and_stride_probe_do_not_calibrate(model_name):
    cfg = config(f"model={model_name}", "qat.fake_quant_start_epoch=1", "qat.observer_freeze_epoch=3")
    model = model_for(cfg)
    smoothers = [layer for layer in model.modules() if isinstance(layer, FixedKernelConv2d)]
    assert bool(smoothers) == model_name.startswith("v9-sr-")
    assert all(isinstance(layer, QATFixedKernelConv2d) for layer in smoothers)
    assert all(list(dict(quantizers(layer))) == ["input_fake_quant"] for layer in smoothers)
    original_quantizers = dict(quantizers(model))
    assert prepare_qat(model, cfg.qat) is model
    assert dict(quantizers(model)) == original_quantizers
    assert model.training
    create_converter(cfg.model.name, model, cfg.model.anchor, list(cfg.image_size), "cpu")
    assert all(
        q.activation_post_process.min_val.numel() == 0 or not torch.isfinite(q.activation_post_process.min_val).all()
        for _, q in quantizers(model)
    )
    model(torch.rand(2, 3, 32, 64))
    assert all(q.fake_quant_enabled.item() == 0 and q.observer_enabled.item() == 1 for _, q in quantizers(model))
    with pytest.raises(ValueError, match="warmup"):
        freeze_for_export(model)
    set_qat_epoch(model, 1)
    assert all(q.fake_quant_enabled.item() == 1 and q.observer_enabled.item() == 1 for _, q in quantizers(model))
    set_qat_epoch(model, 3)
    assert all(q.fake_quant_enabled.item() == 1 and q.observer_enabled.item() == 0 for _, q in quantizers(model))
    assert all(layer.bias.requires_grad for layer in model.modules() if isinstance(layer, QATConv2d))


def test_smoothing_quantizes_input_and_preserves_fixed_kernel():
    source = FixedKernelConv2d(2)
    model = QATFixedKernelConv2d.from_float(source, 1.0).train()
    model.requires_grad_(True)
    x = torch.linspace(-0.3, 1.1, 2 * 7 * 9).reshape(1, 2, 7, 9).requires_grad_()
    model.input_fake_quant.disable_fake_quant()
    torch.testing.assert_close(model(x), source(x), rtol=0, atol=0)
    model.input_fake_quant.disable_observer()
    model.input_fake_quant.enable_fake_quant()
    q = model.input_fake_quant
    quantized = torch.fake_quantize_per_tensor_affine(x, q.scale, q.zero_point, 0, 255)
    assert not torch.equal(x, quantized)
    actual = model(x)
    torch.testing.assert_close(actual, source(quantized), rtol=0, atol=0)
    assert not torch.equal(actual, source(x))
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert not list(model.parameters()) and "weight" not in model.state_dict()
    assert not model.weight.requires_grad and model.weight.grad is None
    torch.testing.assert_close(model.weight, source.weight, rtol=0, atol=0)


@pytest.mark.parametrize("lightning", [False, True])
def test_legacy_sr_qat_without_smoothing_observer_is_rejected(tmp_path, lightning):
    cfg = config("model=v9-sr-t")
    model = calibrated(cfg)
    checkpoint = checkpoint_weights(model)
    state = checkpoint.pop("model_state_dict")
    state = {key: value for key, value in state.items() if ".avg_pool.input_fake_quant." not in key}
    if lightning:
        checkpoint["state_dict"] = {f"model.{key}": value for key, value in state.items()}
    else:
        checkpoint["model_state_dict"] = state
    path = tmp_path / ("legacy.ckpt" if lightning else "legacy.pt")
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match=r"Missing key.*avg_pool.input_fake_quant"):
        create_model(deepcopy(cfg.model), path, 3)


def test_export_requires_matching_checkpoint_and_qdq_flag(tmp_path, monkeypatch):
    cfg = get_cfg(
        ["task=export", "model=v9-t", "weight=false", "image_size=[64,32]", f'task.output={tmp_path / "x.onnx"}']
    )
    cfg.task.qdq = True
    with pytest.raises(ValueError, match="QAT checkpoint"):
        export_model(cfg)
    cfg.task.format = "tflite"
    with pytest.raises(ValueError, match="ONNX-only"):
        export_model(cfg)
    model = calibrated(config())
    cfg.task.format = "onnx"
    cfg.task.qdq = False
    monkeypatch.setattr("yolo.tools.export.create_model", lambda *args, **kwargs: model)
    with pytest.raises(ValueError, match="task.qdq=true"):
        export_model(cfg)
    assert not (tmp_path / "x.onnx").exists()
