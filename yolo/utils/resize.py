"""Resize policy validation and backwards-compatible inverse box geometry."""


def validate_resize_mode(mode):
    if mode not in ("letterbox", "stretch"):
        raise ValueError(f"resize_mode must be 'letterbox' or 'stretch', got {mode!r}")
    return mode


def restore_boxes(boxes, transform):
    """Restore [..., N, 4] boxes from [..., 5 or 6] resize metadata.

    Legacy letterbox uses [scale, left, top, left, top]; stretch uses
    [scale_x, scale_y, left, top, left, top].
    """
    if transform.shape[-1] == 5:
        scale, shift = transform[..., :1], transform[..., 1:]
    elif transform.shape[-1] == 6:
        scale = transform[..., [0, 1, 0, 1]]
        shift = transform[..., 2:]
    else:
        raise ValueError("Resize metadata must contain 5 or 6 elements")
    return (boxes - shift.unsqueeze(-2)) / scale.unsqueeze(-2)
