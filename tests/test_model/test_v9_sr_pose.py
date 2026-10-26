"""SR pose preserves SR features through training, checkpointing and inference."""

from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf

from tests.conftest import get_cfg
from yolo.model.module import AConv2, ADown2, FixedKernelConv2d, PoseDetection
from yolo.model.yolo import create_model
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter
from yolo.utils.model_utils import PostProcess, create_optimizer
from yolo.utils.pose_utils import Pose2Box


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("size", ["t", "s", "m", "c"])
def test_sr_pose_training_checkpoint_and_inference(size, tmp_path):
    cfg = get_cfg(
        [
            f"model=v9-sr-{size}-pose",
            "task=train-pose",
            "dataset=coco-pose",
            "weight=false",
            "image_size=[96,64]",
        ]
    )
    base = get_cfg([f"model=v9-sr-{size}"])
    # In particular, AUX must retain SR smoothing and M's widened fusion paths.
    expected = OmegaConf.to_container(base.model.model)
    for layers in expected.values():
        for layer in layers:
            if "MultiheadDetection" in layer:
                layer["MultiheadPose"] = layer.pop("MultiheadDetection")
    assert OmegaConf.to_container(cfg.model.model) == expected
    assert cfg.model.activation == base.model.activation == "SiLU"

    model = create_model(deepcopy(cfg.model), class_num=1, weight_path=False)
    blocks = [module for module in model.modules() if isinstance(module, (AConv2, ADown2))]
    assert len(blocks) == (5 if size in {"t", "s"} else 8)
    assert all(type(block) is (ADown2 if size == "c" else AConv2) for block in blocks)
    fixed = [block.avg_pool.weight.clone() for block in blocks]
    converter = create_converter(cfg.model.name, model, cfg.model.anchor, cfg.image_size, "cpu")
    assert isinstance(converter, Pose2Box)
    criterion = create_loss_function(cfg, converter)
    optimizer = create_optimizer(model, cfg.task.optimizer)
    model.train()
    images = torch.rand(2, 3, 64, 96)
    targets = torch.zeros(2, 1, 56)
    targets[..., 1:5] = torch.tensor([8, 8, 88, 56])
    targets[..., 5:] = torch.tensor([48, 32, 2]).repeat(17)
    heads = [module for module in model.modules() if isinstance(module, PoseDetection)]
    before = [head.pose_logits.weight.detach().clone() for head in heads]
    output = model(images)
    loss, metrics = criterion(converter(output["AUX"]), converter(output["Main"]), targets)
    assert torch.isfinite(loss)
    assert metrics["Loss/PoseDFLoss"] > 0
    assert metrics["Loss/PoseVisibilityLoss"] > 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for tag in ("Main", "AUX"):
        branch = model.model[model.layer_index[tag] - 1]
        gradients = [head.pose_logits.weight.grad for head in branch.modules() if isinstance(head, PoseDetection)]
        assert sum(gradient.abs().sum() for gradient in gradients) > 0
    optimizer.step()
    assert any(not torch.equal(head.pose_logits.weight, old) for head, old in zip(heads, before))
    assert all(torch.equal(block.avg_pool.weight, old) for block, old in zip(blocks, fixed))

    path = tmp_path / "pose.pt"
    torch.save({"weights": model.model.state_dict(), "pose_config": model.pose_config}, path)
    restored = create_model(deepcopy(cfg.model), class_num=1, weight_path=path).eval()
    restored.require_pose_weights()
    model.eval()
    cfg.task.validation.nms.min_confidence = 0.0
    postprocess = PostProcess(converter, cfg.task.validation.nms)
    with torch.no_grad():
        expected = model(images, shortcut="Main")
        actual = restored(images, shortcut="Main")
        for expected_scale, actual_scale in zip(expected["Main"], actual["Main"]):
            for expected_tensor, actual_tensor in zip(expected_scale, actual_scale):
                torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        predictions = postprocess(actual, image_size=[96, 64])
    assert len(predictions) == 2
    assert all(rows.shape[0] > 0 and rows.shape[1] == 57 and torch.isfinite(rows).all() for rows in predictions)
    restored_fixed = [module.weight for module in restored.modules() if isinstance(module, FixedKernelConv2d)]
    assert all(torch.equal(actual, old) for actual, old in zip(restored_fixed, fixed))
