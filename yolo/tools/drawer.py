import math
import random
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from yolo.config.config import ModelConfig
from yolo.model.yolo import YOLO
from yolo.utils.logger import logger

# Zero-based COCO skeleton; custom keypoint layouts can supply their own edges.
COCO_SKELETON = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


def draw_poses(img, predictions, *, idx2label=None, confidence=0.5, skeleton=None):
    """Draw [class,xyxy,score,K*(x,y,confidence)] rows without changing inputs."""
    rows = predictions[0] if isinstance(predictions, list) or predictions.ndim == 3 else predictions
    img = draw_bboxes(img, rows[:, :6], idx2label=idx2label)
    points = rows[:, 6:].detach().cpu().reshape(len(rows), (rows.shape[1] - 6) // 3, 3)
    count = points.shape[1]
    if skeleton is None:
        skeleton = COCO_SKELETON if count == 17 else ()
    if any(len(edge) != 2 or any(index < 0 or index >= count for index in edge) for edge in skeleton):
        raise ValueError("Skeleton edges must reference valid zero-based keypoint indices.")
    draw = ImageDraw.Draw(img)
    for instance in points.tolist():
        valid = [all(math.isfinite(v) for v in point) and point[2] >= confidence for point in instance]
        for start, end in skeleton:
            if valid[start] and valid[end]:
                draw.line((*instance[start][:2], *instance[end][:2]), fill=(0, 220, 100), width=2)
        for index, (x, y, _) in enumerate(instance):
            if valid[index]:
                x, y = round(x), round(y)
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 100, 30))
    return img


def draw_bboxes(
    img: Union[Image.Image, torch.Tensor],
    bboxes: List[List[Union[int, float]]],
    *,
    idx2label: Optional[list] = None,
):
    """
    Draw bounding boxes on an image.

    Args:
    - img (PIL Image or torch.Tensor): Image on which to draw the bounding boxes.
    - bboxes (List of Lists/Tensors): Bounding boxes with [class_id, x_min, y_min, x_max, y_max],
      where coordinates are in image pixels, optionally followed by confidence.
    """
    # Convert tensor image to PIL Image if necessary
    if isinstance(img, torch.Tensor):
        if img.dim() > 3:
            logger.warning("🔍 >3 dimension tensor detected, using the 0-idx image.")
            img = img[0]
        img = to_pil_image(img)

    if isinstance(bboxes, list) or bboxes.ndim == 3:
        bboxes = bboxes[0]

    img = img.copy()
    label_size = img.size[1] / 30
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype("arial.ttf", int(label_size))
    except IOError:
        font = ImageFont.load_default(int(label_size))

    for bbox in bboxes:
        class_id, x_min, y_min, x_max, y_max, *conf = [float(val) for val in bbox]
        # Pillow computes corner radii before rounding fractional coordinates.
        # For small boxes this can invert its internal outline rectangles even
        # when x_min <= x_max and y_min <= y_max. Rasterize first, leaving the
        # original prediction coordinates unchanged.
        x_min, x_max = round(min(x_min, x_max)), round(max(x_min, x_max))
        y_min, y_max = round(min(y_min, y_max)), round(max(y_min, y_max))
        bbox = [(x_min, y_min), (x_max, y_max)]

        color_rng = random.Random(int(class_id))
        color_map = tuple(color_rng.randint(0, 200) for _ in range(3))

        draw.rounded_rectangle(bbox, outline=(*color_map, 200), radius=5, width=2)
        draw.rounded_rectangle(bbox, fill=(*color_map, 100), radius=5)

        class_text = str(idx2label[int(class_id)] if idx2label else int(class_id))
        label_text = f"{class_text}" + (f" {conf[0]: .0%}" if conf else "")

        text_bbox = font.getbbox(label_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = round((text_bbox[3] - text_bbox[1]) * 1.5)

        text_background = [(x_min, y_min), (x_min + text_width, y_min + text_height)]
        draw.rounded_rectangle(text_background, fill=(*color_map, 175), radius=2)
        draw.text((x_min, y_min), label_text, fill="white", font=font)

    return img


def draw_model(*, model_cfg: ModelConfig = None, model: YOLO = None, v7_base=False):
    from graphviz import Digraph

    if model_cfg:
        from yolo.model.yolo import create_model

        model = create_model(model_cfg)
    elif model is None:
        raise ValueError("Drawing Object is None")

    model_size = len(model.model) + 1
    model_mat = np.zeros((model_size, model_size), dtype=bool)

    layer_name = ["INPUT"]
    for idx, layer in enumerate(model.model, start=1):
        layer_name.append(str(type(layer)).split(".")[-1][:-2])
        if layer.tags is not None:
            layer_name[-1] = f"{layer.tags}-{layer_name[-1]}"
        if isinstance(layer.source, int):
            source = layer.source + (layer.source < 0) * idx
            model_mat[source, idx] = True
        else:
            for source in layer.source:
                source = source + (source < 0) * idx
                model_mat[source, idx] = True

    pattern_mat = []
    if v7_base:
        pattern_list = [("ELAN", 8, 3), ("ELAN", 8, 55), ("MP", 5, 11)]
        for name, size, position in pattern_list:
            pattern_mat.append(
                (name, size, model_mat[position : position + size, position + 1 : position + 1 + size].copy())
            )

    dot = Digraph(comment="Model Flow Chart")
    node_idx = 0

    for idx in range(model_size):
        for jdx in range(idx, model_size - 7):
            for name, size, pattern in pattern_mat:
                if (model_mat[idx : idx + size, jdx : jdx + size] == pattern).all():
                    layer_name[idx] = name
                    model_mat[idx : idx + size, jdx : jdx + size] = False
                    model_mat[idx, idx + size] = True
        dot.node(str(idx), f"{layer_name[idx]}")
        node_idx += 1
        for jdx in range(idx, model_size):
            if model_mat[idx, jdx]:
                dot.edge(str(idx), str(jdx))
    try:
        dot.render("Model-arch", format="png", cleanup=True)
        logger.info(":artist_palette: Drawing Model Architecture at Model-arch.png")
    except:
        logger.warning(":warning: Could not find graphviz backend, continue without drawing the model architecture")
