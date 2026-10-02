from pathlib import Path
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir

import yolo.lazy as lazy


@pytest.mark.parametrize("image_size,expected", [("640", [640, 640]), ("[640,480]", [640, 480])])
@pytest.mark.parametrize("task", ["export", "inference"])
def test_image_size_is_normalized_before_dispatch(monkeypatch, image_size, expected, task):
    with initialize_config_dir(config_dir=str(Path(lazy.__file__).parent / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"task={task}", f"image_size={image_size}", "weight=model.onnx"])
    handler = Mock()
    target = "yolo.tools.export.export_model" if task == "export" else "yolo.tools.onnx_runner.run_onnx_inference"
    monkeypatch.setattr(target, handler)
    monkeypatch.setattr(lazy, "set_seed", lambda seed: None)

    lazy.main.__wrapped__(cfg)

    handler.assert_called_once_with(cfg)
    assert list(cfg.image_size) == expected
    if task == "inference":
        assert list(cfg.task.data.image_size) == expected
