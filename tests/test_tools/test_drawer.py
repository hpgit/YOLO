import sys
from pathlib import Path

import pytest
from PIL import Image
from torch import tensor

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import YOLO
from yolo.tools.drawer import draw_bboxes, draw_model


def test_draw_model_by_config(train_cfg: Config):
    """Test the drawing of a model based on a configuration."""
    draw_model(model_cfg=train_cfg.model)


def test_draw_model_by_model(model: YOLO):
    """Test the drawing of a YOLO model."""
    draw_model(model=model)


def test_draw_bboxes():
    """Test drawing bounding boxes on an image."""
    predictions = tensor([[0, 60, 60, 160, 160, 0.5], [0, 40, 40, 120, 120, 0.5]])
    pil_image = Image.open("tests/data/images/train/000000050725.jpg")
    draw_bboxes(pil_image, [predictions])


@pytest.mark.parametrize(
    "coordinates",
    [
        # Valid, ordered fractional boxes that trigger Pillow's internal outline
        # rectangles to invert after it rounds the corners (Pillow 12.3.0).
        [9.155748, 34.635998, 16.435223, 40.519670],
        [34.635998, 9.155748, 40.519670, 16.435223],
        [10.1, 20.1, 10.2, 20.2],  # Subpixel box collapses to a pixel.
        [10.1, 20.1, 10.2, 50.2],  # Very thin box.
        [-4.2, -3.8, 5.4, 6.7],  # Partially outside the image.
        [16.435223, 40.519670, 9.155748, 34.635998],  # Reversed endpoints.
    ],
)
def test_draw_fractional_boxes_without_changing_predictions(coordinates):
    image = Image.new("RGB", (128, 128))
    original_image = image.tobytes()
    predictions = tensor([[0, *coordinates, 0.9]])
    original_predictions = predictions.clone()

    rendered = draw_bboxes(image, [predictions], idx2label=["object"])

    assert rendered.size == image.size
    assert rendered.tobytes() != original_image
    assert image.tobytes() == original_image
    assert predictions.equal(original_predictions)
