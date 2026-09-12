"""Opt-in checks against a separately provided original YOLOv9 checkout."""

import os
from pathlib import Path

import pytest

from scripts.verify_yolov9_augmentation import load_reference, run_checks


@pytest.mark.skipif(
    not os.environ.get("YOLOV9_REFERENCE"),
    reason="Set YOLOV9_REFERENCE to the pinned external YOLOv9 checkout",
)
def test_original_yolov9_augmentation_parity():
    reference, hashes = load_reference(Path(os.environ["YOLOV9_REFERENCE"]))
    checks = run_checks(reference, seeds=8)
    assert len(hashes) == 4
    assert checks
    assert all(check["passed"] for check in checks), checks
