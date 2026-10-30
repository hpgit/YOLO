"""Joint-aware single-image pose augmentation using the worker's Torch RNG."""

import math

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def pose_affine(image, targets, *, degrees, translate, scale, num_keypoints=17):
    """Apply one similarity transform to pixels, boxes, joints, and instance area.

    ``translate`` is in pixels. The last three target columns contain instance
    area and final resize gains. Area tracks the transformed full instance;
    cropping does not replace segmentation area with clipped bounding-box area.
    """
    width, height = image.size
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians) * scale, math.sin(radians) * scale
    linear = torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.float64)
    center = torch.tensor([width / 2, height / 2], dtype=torch.float64)
    shift = center - linear @ center + torch.tensor(translate, dtype=torch.float64)
    inverse = torch.linalg.inv(linear)
    inverse_shift = -inverse @ shift
    coefficients = torch.cat((inverse, inverse_shift[:, None]), dim=1).flatten().tolist()
    image = image.transform(
        image.size,
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    if not targets.numel():
        return image, targets

    linear, shift = linear.to(targets), shift.to(targets)
    # Rotation needs all four box corners, not just the diagonal pair.
    corners = targets[:, [1, 2, 3, 2, 3, 4, 1, 4]].reshape(-1, 4, 2)
    corners = corners @ linear.T + shift
    lower, upper = corners.amin(1), corners.amax(1)
    full_box_area = (upper - lower).prod(-1)
    lower[:, 0].clamp_(0, width)
    lower[:, 1].clamp_(0, height)
    upper[:, 0].clamp_(0, width)
    upper[:, 1].clamp_(0, height)
    dimensions = upper - lower
    retained_fraction = dimensions.prod(-1) / full_box_area.clamp_min(1e-8)
    keep = (dimensions > 1).all(-1) & (retained_fraction >= 0.2)
    targets[:, 1:3], targets[:, 3:5] = lower, upper

    joint_end = 5 + 3 * num_keypoints
    joints = targets[:, 5:joint_end].reshape(-1, num_keypoints, 3)
    xy = joints[..., :2] @ linear.T + shift
    inside = (xy[..., 0] >= 0) & (xy[..., 0] <= width) & (xy[..., 1] >= 0) & (xy[..., 1] <= height)
    labeled = (joints[..., 2] > 0) & inside
    joints[..., :2] = torch.where(labeled[..., None], xy, 0)
    joints[..., 2] = torch.where(labeled, joints[..., 2], 0)
    targets[:, joint_end] *= scale**2
    return image, targets[keep]


def random_pose_affine(image, targets, options, num_keypoints=17):
    """Sample isotropic scale, rotation and x/y translation without shearing."""
    if not any(options[key] for key in ("degrees", "translate", "scale")):
        return image, targets
    samples = torch.rand(4).mul(2).sub(1).tolist()
    return pose_affine(
        image,
        targets,
        degrees=samples[0] * options["degrees"],
        scale=1 + samples[1] * options["scale"],
        translate=(samples[2] * options["translate"] * image.width, samples[3] * options["translate"] * image.height),
        num_keypoints=num_keypoints,
    )


def pose_color_jitter(image, options):
    """Perturb hue, saturation and value (brightness) without touching labels."""
    for name, transform, offset in (
        ("hsv_h", TF.adjust_hue, 0),
        ("hsv_s", TF.adjust_saturation, 1),
        ("hsv_v", TF.adjust_brightness, 1),
    ):
        magnitude = options[name]
        if magnitude:
            image = transform(image, offset + (torch.rand(()).item() * 2 - 1) * magnitude)
    return image
