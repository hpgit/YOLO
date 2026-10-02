"""Compare augmentation to a separately supplied, pinned original YOLOv9 checkout.

The reference implementation is read from its Git object database at runtime;
no original YOLOv9 source is included in this repository. This is a synthetic
augmentation check, not training or accuracy reproduction.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE_COMMIT = "5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff"


def load_reference(path):
    """Execute only the named reference definitions, with minimal dependencies."""
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != REFERENCE_COMMIT:
        raise ValueError(f"Expected reference {REFERENCE_COMMIT}, got {revision}")
    namespace = {
        "np": np,
        "cv2": cv2,
        "torch": torch,
        "math": math,
        "random": random,
        # These helpers affect only upstream logging/version validation. The
        # implementation requires 1.3.1 independently; augmentation is unmodified.
        "LOGGER": SimpleNamespace(info=lambda *_: None),
        "colorstr": lambda *_: "",
        "check_version": lambda *_args, **_kwargs: True,
    }
    selections = {
        "utils/general.py": (
            "xywhn2xyxy",
            "xyxy2xywhn",
            "xyxy2xywh",
            "xyn2xy",
            "clip_boxes",
            "resample_segments",
            "segment2box",
        ),
        "utils/metrics.py": ("bbox_ioa",),
        "utils/augmentations.py": (
            "augment_hsv",
            "letterbox",
            "random_perspective",
            "copy_paste",
            "mixup",
            "box_candidates",
            "Albumentations",
        ),
        "utils/dataloaders.py": ("LoadImagesAndLabels",),
    }
    hashes = {}
    for filename, names in selections.items():
        source = subprocess.check_output(["git", "-C", str(path), "show", f"{REFERENCE_COMMIT}:{filename}"])
        hashes[filename] = hashlib.sha256(source).hexdigest()
        nodes = []
        for node in ast.parse(source, filename=filename).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                if isinstance(node, ast.ClassDef) and node.name == "LoadImagesAndLabels":
                    node.bases = []
                    node.body = [
                        item
                        for item in node.body
                        if isinstance(item, ast.FunctionDef)
                        and item.name in ("__getitem__", "load_image", "load_mosaic")
                    ]
                nodes.append(node)
        if {node.name for node in nodes} != set(names):
            raise ValueError(f"Reference definitions missing in {filename}")
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(module, filename, "exec"), namespace)
    return namespace, hashes


def seeded_call(function, seed, *args, **kwargs):
    random.seed(seed)
    np.random.seed(seed)
    result = function(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    return result, (random.getstate(), np.random.get_state())


def compare_arrays(expected, actual, tolerance=0):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape:
        return {"passed": False, "expected_shape": list(expected.shape), "actual_shape": list(actual.shape)}
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    error = float(difference.max()) if difference.size else 0.0
    return {"passed": bool(error <= tolerance), "max_abs_error": error, "shape": list(expected.shape)}


def compare_rng(expected, actual):
    py_equal = expected[0] == actual[0]
    np_equal = (
        expected[1][0] == actual[1][0]
        and np.array_equal(expected[1][1], actual[1][1])
        and expected[1][2:] == actual[1][2:]
    )
    return bool(py_equal and np_equal)


def fixture():
    """Color texture and asymmetrical objects expose color/order/crop errors."""
    rng = np.random.default_rng(7301)
    image = rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)
    labels = np.array(
        [[0, 12, 15, 65, 95], [1, 91, 40, 117, 156], [2, 173, 72, 244, 180]],
        dtype=np.float32,
    )
    segments = [
        np.array([[12, 15], [65, 28], [57, 95], [18, 82]], dtype=np.float32),
        np.array([[91, 40], [117, 52], [111, 156], [96, 140]], dtype=np.float32),
        np.array([[173, 72], [244, 81], [239, 180], [182, 165]], dtype=np.float32),
    ]
    return image, labels, segments


def check_case(name, reference_call, implementation_call, seeds, tolerances):
    """Check outputs and final RNG state, retaining failures for diagnosis."""
    failures = []
    maximums = [0.0] * len(tolerances)
    for seed in range(seeds):
        expected, expected_rng = seeded_call(reference_call, seed)
        actual, actual_rng = seeded_call(implementation_call, seed)
        components = [
            compare_arrays(left, right, tolerance) for left, right, tolerance in zip(expected, actual, tolerances)
        ]
        if len(expected) != len(tolerances) or len(actual) != len(tolerances):
            raise ValueError(f"Wrong output arity for {name}")
        for index, component in enumerate(components):
            maximums[index] = max(maximums[index], component.get("max_abs_error", 0))
        rng_equal = compare_rng(expected_rng, actual_rng)
        if not all(component["passed"] for component in components) or not rng_equal:
            failures.append({"seed": seed, "components": components, "rng_equal": rng_equal})
    return {
        "name": name,
        "passed": not failures,
        "maximum_abs_errors": maximums,
        "tolerances": list(tolerances),
        "seeds": seeds,
        "failed_seeds": failures,
    }


def run_checks(reference, seeds=32):
    from yolo.tools import yolov9_augmentation as implementation

    image, labels, segments = fixture()
    checks = []

    def hsv_reference():
        output = image.copy()
        reference["augment_hsv"](output, 0.015, 0.7, 0.4)
        return (output,)

    def hsv_implementation():
        output = image[:, :, ::-1].copy()
        implementation._augment_hsv(output, 0.015, 0.7, 0.4)
        return (output[:, :, ::-1],)

    checks.append(check_case("hsv", hsv_reference, hsv_implementation, seeds, (0,)))
    for has_segments in (False, True):
        for perspective in (0.0, 0.0005):
            selected_segments = segments if has_segments else []

            def expected_geometry():
                return reference["random_perspective"](
                    image.copy(),
                    labels.copy(),
                    copy.deepcopy(selected_segments),
                    degrees=0,
                    translate=0.1,
                    scale=0.9,
                    shear=0,
                    perspective=perspective,
                    border=(-32, -32),
                )

            def actual_geometry():
                aligned = selected_segments or [np.zeros((0, 2), dtype=np.float32) for _ in labels]
                return implementation._random_perspective(
                    image.copy(),
                    labels.copy(),
                    copy.deepcopy(aligned),
                    degrees=0,
                    translate=0.1,
                    scale_gain=0.9,
                    shear=0,
                    perspective=perspective,
                    border=(-32, -32),
                )

            name = f"geometry_{'segments' if has_segments else 'boxes'}_perspective_{perspective}"
            checks.append(check_case(name, expected_geometry, actual_geometry, seeds, (0, 0.0001)))

    for probability in (0.3, 1.0):

        def expected_copy():
            result = reference["copy_paste"](image.copy(), labels.copy(), copy.deepcopy(segments), p=probability)
            return result[0], result[1], np.stack(result[2])

        def actual_copy():
            result = implementation._copy_paste(
                image.copy(), labels.copy(), copy.deepcopy(segments), probability=probability
            )
            return result[0], result[1], np.stack(result[2])

        checks.append(check_case(f"copy_paste_{probability}", expected_copy, actual_copy, seeds, (0, 0, 0)))

    with tempfile.TemporaryDirectory(prefix="yolov9-augmentation-parity-") as directory:
        dataset = reference["LoadImagesAndLabels"]()
        dataset.img_size = 128
        dataset.augment = True
        dataset.im_files = []
        dataset.npy_files = []
        dataset.ims = [None] * 4
        source_images = []
        for index, (height, width) in enumerate(((192, 256), (241, 131), (113, 229), (171, 171))):
            source = np.random.default_rng(100 + index).integers(0, 256, (height, width, 3), dtype=np.uint8)
            path = Path(directory) / f"{index}.png"
            cv2.imwrite(str(path), source)
            dataset.im_files.append(str(path))
            dataset.npy_files.append(path.with_suffix(".npy"))
            source_images.append(source)
            checks.append(
                check_case(
                    f"resize_{height}x{width}",
                    lambda: (dataset.load_image(index)[0],),
                    lambda: (implementation._resize_longest(source, 128),),
                    1,
                    (0,),
                )
            )
        for albumentations in (False, True):
            checks.extend(check_pipelines(reference, implementation, dataset, source_images, seeds, albumentations))
    checks.extend(check_albumentations(reference, implementation, seeds))
    return checks


def check_pipelines(reference, implementation, dataset, source_images, seeds, albumentations):
    """Use original dataset methods for geometry, mosaic, MixUp and full order."""
    dataset.n = len(source_images)
    dataset.indices = list(range(dataset.n))
    dataset.mosaic_border = [-64, -64]
    dataset.mosaic = True
    dataset.rect = False
    dataset.albumentations = (
        reference["Albumentations"](128) if albumentations else lambda image, labels: (image, labels)
    )
    if albumentations and dataset.albumentations.transform is None:
        raise RuntimeError("Original Albumentations failed to initialize; cannot verify enabled-stage parity")
    normal_xywh = np.array([[0, 0.1875, 0.25, 0.25, 0.375], [1, 0.6875, 0.625, 0.25, 0.375]], dtype=np.float32)
    normal_xyxy = reference["xywhn2xyxy"](normal_xywh[:, 1:], 1, 1)
    source_boxes = np.column_stack((normal_xywh[:, 0], normal_xyxy))
    source_segments = [
        np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32) for x1, y1, x2, y2 in normal_xyxy
    ]
    dataset.labels = [normal_xywh.copy() for _ in source_images]
    checks = []
    for label_mode in ("boxes", "segments", "empty"):
        use_segments = label_mode == "segments"
        selected_boxes = source_boxes if label_mode != "empty" else np.zeros((0, 5), dtype=np.float32)
        dataset.labels = [normal_xywh.copy() if label_mode != "empty" else selected_boxes.copy() for _ in source_images]
        dataset.segments = [copy.deepcopy(source_segments) if use_segments else [] for _ in source_images]
        for mosaic, mixup in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 0.15)):
            augmentation = implementation.YOLOv9Augmentation(
                128, mosaic=mosaic, mixup=mixup, albumentations=albumentations
            )
            dataset.hyp = augmentation.hyp.copy()

            def expected_pipeline():
                output_image, output_labels, _, _ = dataset[0]
                labels_xyxy = output_labels[:, 1:].numpy().copy()
                labels_xyxy[:, 1:] = reference["xywhn2xyxy"](labels_xyxy[:, 1:], 1, 1)
                return output_image.numpy(), labels_xyxy

            def actual_pipeline():
                def raw_sample(index):
                    return (
                        source_images[index][:, :, ::-1].copy(),
                        selected_boxes.copy(),
                        copy.deepcopy(source_segments) if use_segments else [],
                    )

                def sample(count):
                    if count == 1:
                        return raw_sample(random.randint(0, dataset.n - 1))
                    return [raw_sample(index) for index in random.choices(dataset.indices, k=count)]

                output_image, output_labels, _ = augmentation(
                    source_images[0][:, :, ::-1].copy(),
                    selected_boxes.copy(),
                    copy.deepcopy(source_segments) if use_segments else [],
                    sample,
                )
                pixels = output_image.mul(255).round().to(torch.uint8).numpy()
                return pixels, output_labels.numpy()

            name = f"pipeline_{label_mode}_mosaic_{mosaic}_mixup_{mixup}_albumentations_{albumentations}"
            checks.append(check_case(name, expected_pipeline, actual_pipeline, seeds, (0, 0.00001)))
    return checks


def check_albumentations(reference, implementation, seeds):
    image, labels, _ = fixture()
    labels[:, [1, 3]] /= image.shape[1]
    labels[:, [2, 4]] /= image.shape[0]
    expected_labels = labels.copy()
    expected_labels[:, 1:] = reference["xyxy2xywh"](labels[:, 1:])
    checks = []
    for name in ("Blur", "MedianBlur", "ToGray", "CLAHE"):
        original = reference["Albumentations"](128)
        augmented = implementation.YOLOv9Augmentation(128, albumentations=True)
        if original.transform is None:
            raise RuntimeError("Original Albumentations failed to initialize")
        for compose in (original.transform, augmented._albumentations):
            for transform in compose.transforms:
                transform.p = float(type(transform).__name__ == name)

        def expected_stage():
            result_image, result_labels = original(image.copy(), expected_labels.copy())
            result_labels[:, 1:] = reference["xywhn2xyxy"](result_labels[:, 1:], 1, 1)
            return result_image, result_labels

        def actual_stage():
            result_image, result_labels = augmented._apply_albumentations(image[:, :, ::-1].copy(), labels.copy())
            return result_image[:, :, ::-1], result_labels

        checks.append(check_case(f"albumentations_forced_{name}", expected_stage, actual_stage, seeds, (0, 0.000001)))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--seeds", type=int, default=32)
    args = parser.parse_args()
    try:
        if args.seeds < 1:
            raise ValueError("--seeds must be positive")
        reference, hashes = load_reference(args.reference)
        checks = run_checks(reference, args.seeds)
        report = {
            "passed": all(check["passed"] for check in checks),
            "reference_commit": REFERENCE_COMMIT,
            "reference_source_sha256": hashes,
            "versions": {
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "albumentations": version("albumentations"),
            },
            "seeds_per_case": args.seeds,
            "checks": checks,
            "scope": "Synthetic image/label/RNG parity; Albumentations enabled and disabled, plus each active photometric transform forced; no training or accuracy claim.",
        }
    except Exception as error:
        report = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
