#!/usr/bin/env python3
"""Portable YOLO ONNX inference; this file can be copied out of the repository.

Requires numpy, Pillow and onnxruntime (opencv-python-headless for video/webcam).
Run: python onnx_inference.py --model model.onnx --source image.jpg --output results
New exports carry decoder and NMS-free postprocessing metadata. For older exports supply --class-num and,
if needed, --output-format, --reg-max and --strides. No PyTorch or YOLO imports.
"""

import argparse
import json
from contextlib import ExitStack
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def class_aware_nms(boxes, scores, confidence=0.5, iou_threshold=0.5, max_detections=300):
    """Multi-label NMS -> float32 rows [class_id, x1, y1, x2, y2, score].

    Scores are already probabilities: do not apply sigmoid or objectness again.
    Suppress within each class, then sort all surviving detections by score.
    """
    candidates = []
    valid_boxes = np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
    for class_id in range(scores.shape[1]):
        indices = np.flatnonzero(valid_boxes & np.isfinite(scores[:, class_id]) & (scores[:, class_id] > confidence))
        order = indices[np.argsort(-scores[indices, class_id], kind="stable")]
        kept = []
        while order.size and len(kept) < max_detections:
            index = order[0]
            kept.append(index)
            remaining = order[1:]
            intersection = np.maximum(
                np.minimum(boxes[index, 2:], boxes[remaining, 2:]) - np.maximum(boxes[index, :2], boxes[remaining, :2]),
                0,
            ).prod(axis=1)
            area = (boxes[index, 2:] - boxes[index, :2]).prod()
            other_area = (boxes[remaining, 2:] - boxes[remaining, :2]).prod(axis=1)
            union = area + other_area - intersection
            overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = remaining[overlap <= iou_threshold]
        if kept:
            candidates.append(np.column_stack((np.full(len(kept), class_id), boxes[kept], scores[kept, class_id])))
    if not candidates:
        return np.empty((0, 6), dtype=np.float32)
    detections = np.concatenate(candidates).astype(np.float32)
    return detections[np.argsort(-detections[:, 5], kind="stable")[:max_detections]]


def nms_free_topk(boxes, scores, confidence=0.5, max_detections=300):
    """Best class per anchor and stable score top-k; never suppress overlaps.

    Scores are probabilities from a one-to-one trained head. Returns float32
    rows [class_id, x1, y1, x2, y2, score] in descending confidence order.
    """
    if not 0 <= confidence <= 1 or max_detections < 1:
        raise ValueError("Confidence must be in [0,1] and max_detections must be positive.")
    class_ids = scores.argmax(axis=-1)
    best_scores = scores[np.arange(len(scores)), class_ids]
    valid = (
        np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        & np.isfinite(best_scores)
        & (best_scores > confidence)
    )
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(-best_scores[indices], kind="stable")[:max_detections]]
    return np.column_stack((class_ids[indices], boxes[indices], best_scores[indices])).astype(np.float32)


class ONNXDetector:
    """Float32 NCHW ONNX runtime and CPU NumPy postprocessing.

    predict accepts PIL images and returns one (N, 6) array per image in original
    image pixels. __call__ accepts preprocessed RGB [B,3,H,W] float32 arrays and
    returns decoded [B,N,4+C] before filtering. Metadata takes precedence over
    decoder fallback arguments. Fixed batch models require exactly that batch.
    """

    def __init__(
        self,
        model,
        *,
        providers=None,
        class_num=None,
        output_format="auto",
        reg_max=16,
        strides=(8, 16, 32),
        confidence=0.5,
        iou_threshold=0.5,
        max_detections=300,
        threads=0,
        nms_free=False,
    ):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError("Install runtime dependencies: pip install numpy Pillow onnxruntime") from error
        if not 0 <= confidence <= 1 or not 0 <= iou_threshold <= 1 or max_detections < 1:
            raise ValueError("Confidence/IoU must be in [0,1] and max_detections must be positive.")
        if not Path(model).is_file():
            raise FileNotFoundError(model)
        providers = list(providers or ["CPUExecutionProvider"])
        missing = set(providers) - set(ort.get_available_providers())
        if missing:
            raise ValueError(
                f"Unavailable ONNX providers: {sorted(missing)}; available: {ort.get_available_providers()}"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one input and one output from task=export; raw-head ONNX is unsupported.")
        self.input, self.output = inputs[0], outputs[0]
        shape = self.input.shape
        if (
            self.input.type != "tensor(float)"
            or len(shape) != 4
            or shape[1] != 3
            or any(not isinstance(n, int) or n <= 0 for n in shape[2:])
        ):
            raise ValueError("Expected float32 RGB [B,3,H,W] with fixed positive spatial dimensions.")
        if self.output.type != "tensor(float)" or len(self.output.shape) != 3:
            raise ValueError("Expected one float32 [B,N,4*reg_max+C] or [B,N,4+C] output.")
        self.batch_size = shape[0] if isinstance(shape[0], int) else None
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("ONNX batch size must be positive.")
        self.height, self.width = shape[2:]
        props = self.session.get_modelmeta().custom_metadata_map
        metadata = json.loads(props.get("yolo.inference", "{}"))
        if metadata and metadata.get("version") != 1:
            raise ValueError("Unsupported yolo.inference metadata version.")
        self.nms_free = metadata.get("nms_free", nms_free)
        if not isinstance(self.nms_free, bool):
            raise ValueError("nms_free must be a boolean in ONNX metadata or fallback arguments.")
        postprocess = metadata.get("postprocess", "topk" if self.nms_free else "nms")
        if postprocess not in ("topk", "nms"):
            raise ValueError("Unsupported ONNX postprocess method; expected topk or nms.")
        if "nms_free" in metadata and (postprocess == "topk") != self.nms_free:
            raise ValueError("ONNX nms_free and postprocess metadata disagree.")
        self.nms_free = postprocess == "topk"
        self.class_num = metadata.get("class_num", class_num)
        if not isinstance(self.class_num, int) or self.class_num < 1:
            raise ValueError("ONNX has no decoder metadata; supply class_num / --class-num from the export dataset.")
        self.class_names = metadata.get("class_names", [])
        if self.class_names and len(self.class_names) != self.class_num:
            raise ValueError("ONNX class_names length does not match class_num.")
        self.reg_max = metadata.get("reg_max", reg_max)
        self.output_format = metadata.get("output_format", output_format)
        channels = self.output.shape[-1]
        if self.output_format == "auto":
            if channels == self.class_num + 4:
                self.output_format = "xyxy"
            elif channels == self.class_num + 4 * self.reg_max:
                self.output_format = "dfl"
            else:
                raise ValueError("Cannot identify ONNX output format; check class_num, reg_max and output_format.")
        if self.output_format not in ("dfl", "xyxy"):
            raise ValueError("output_format must be auto, dfl or xyxy.")
        self.channels = self.class_num + (4 * self.reg_max if self.output_format == "dfl" else 4)
        if isinstance(channels, int) and channels != self.channels:
            raise ValueError(f"Output channels {channels} do not match decoder contract {self.channels}.")
        if self.output_format == "dfl":
            if not isinstance(self.reg_max, int) or self.reg_max < 2:
                raise ValueError("DFL reg_max must be an integer >= 2.")
            grids, scales = [], []
            strides = metadata.get("strides", strides)
            if not strides:
                raise ValueError("DFL requires head strides in export order.")
            for stride in strides:
                if not isinstance(stride, int) or stride <= 0 or self.width % stride or self.height % stride:
                    raise ValueError("DFL strides must be positive integers dividing the input dimensions.")
                x, y = np.meshgrid(np.arange(self.width // stride), np.arange(self.height // stride))
                grids.append(np.stack((x, y), axis=-1).reshape(-1, 2) * stride + stride // 2)
                scales.append(np.full(x.size, stride))
            self.anchors = np.concatenate(grids).astype(np.float32)
            self.scales = np.concatenate(scales).astype(np.float32)[None, :, None]
            count = self.output.shape[1]
            if isinstance(count, int) and count != len(self.anchors):
                raise ValueError("DFL candidate count does not match strides and input dimensions.")
        self.confidence, self.iou_threshold, self.max_detections = confidence, iou_threshold, max_detections

    def decode(self, predictions):
        """Decode probabilities without applying another sigmoid or softmax."""
        if predictions.ndim != 3 or predictions.shape[-1] != self.channels:
            raise ValueError(f"Unexpected output shape {predictions.shape}; expected [B,N,{self.channels}].")
        if self.output_format == "xyxy":
            return predictions
        batch, count, _ = predictions.shape
        if count != len(self.anchors):
            raise ValueError("DFL candidate count does not match strides and input dimensions.")
        probabilities = predictions[..., : 4 * self.reg_max].reshape(batch, count, 4, self.reg_max)
        distances = (probabilities @ np.arange(self.reg_max, dtype=np.float32)) * self.scales
        boxes = np.concatenate((self.anchors - distances[..., :2], self.anchors + distances[..., 2:]), axis=-1)
        return np.concatenate((boxes, predictions[..., 4 * self.reg_max :]), axis=-1)

    def __call__(self, images):
        images = np.asarray(images, dtype=np.float32)
        if (
            images.ndim != 4
            or images.shape[1:] != (3, self.height, self.width)
            or images.shape[0] < 1
            or self.batch_size is not None
            and images.shape[0] != self.batch_size
        ):
            raise ValueError(f"Input batch must match ONNX shape {self.input.shape}; received {images.shape}.")
        predictions = self.session.run([self.output.name], {self.input.name: np.ascontiguousarray(images)})[0]
        return self.decode(predictions)

    def preprocess(self, image):
        image = image.convert("RGB")
        width, height = image.size
        scale = min(self.width / width, self.height / height)
        resized_width, resized_height = max(1, int(width * scale)), max(1, int(height * scale))
        left, top = (self.width - resized_width) // 2, (self.height - resized_height) // 2
        padded = Image.new("RGB", (self.width, self.height), (114, 114, 114))
        padded.paste(image.resize((resized_width, resized_height), Image.Resampling.LANCZOS), (left, top))
        tensor = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1) / 255.0
        # Use actual rounded resize dimensions for exact inverse coordinates.
        transform = (resized_width / width, resized_height / height, left, top, width, height)
        return tensor, transform

    def predict(self, images):
        """PIL image or list of PIL images -> list of original-coordinate detections."""
        if isinstance(images, Image.Image):
            images = [images]
        if not images:
            return []
        tensors, transforms = zip(*(self.preprocess(image) for image in images))
        results = []
        for prediction, transform in zip(self(np.stack(tensors)), transforms):
            if self.nms_free:
                detections = nms_free_topk(prediction[:, :4], prediction[:, 4:], self.confidence, self.max_detections)
            else:
                detections = class_aware_nms(
                    prediction[:, :4], prediction[:, 4:], self.confidence, self.iou_threshold, self.max_detections
                )
            sx, sy, left, top, width, height = transform
            detections[:, [1, 3]] = np.clip((detections[:, [1, 3]] - left) / sx, 0, width)
            detections[:, [2, 4]] = np.clip((detections[:, [2, 4]] - top) / sy, 0, height)
            results.append(detections[(detections[:, 3:5] > detections[:, 1:3]).all(axis=1)])
        return results


def iter_images(source):
    """Yield (source identifier, RGB PIL image); video support is optional."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    path = Path(str(source))
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in extensions:
                with Image.open(child) as image:
                    yield str(child), image.convert("RGB")
    elif path.suffix.lower() in extensions:
        with Image.open(path) as image:
            yield str(path), image.convert("RGB")
    else:
        webcam = isinstance(source, int) or str(source).isdigit()
        if not webcam and not path.is_file() and "://" not in str(source):
            raise FileNotFoundError(source)
        try:
            import cv2
        except ImportError as error:
            raise ImportError("Video/webcam inference requires: pip install opencv-python-headless") from error
        capture = cv2.VideoCapture(int(source) if webcam else str(source))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video source: {source}")
            index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                yield f"{source}#{index}", Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                index += 1
        finally:
            capture.release()


def draw_detections(image, detections, class_names):
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for class_id, x1, y1, x2, y2, score in detections:
        class_id = int(class_id)
        label = class_names[class_id] if class_names else str(class_id)
        color = ((37 * class_id + 80) % 256, (17 * class_id + 180) % 256, (97 * class_id + 120) % 256)
        draw.rectangle((round(x1), round(y1), round(x2), round(y2)), outline=color, width=2)
        draw.text((round(x1), max(0, round(y1) - 12)), f"{label} {score:.2f}", fill=color)
    return image


def run_inference(detector, source, output=None):
    """Consume the entire source, saving per-frame JPEGs and predictions.jsonl.

    Fixed batches are filled by repeating the last image at EOF; padded results
    are discarded. No frames or predictions are retained after each batch.
    """
    output = Path(output) if output is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    count = 0
    stream = iter_images(source)
    try:
        with ExitStack() as stack:
            records = (
                stack.enter_context((output / "predictions.jsonl").open("w", encoding="utf-8")) if output else None
            )
            while True:
                batch = list(islice(stream, detector.batch_size or 1))
                if not batch:
                    break
                images = [image for _, image in batch]
                images += [images[-1]] * ((detector.batch_size or len(images)) - len(images))
                predictions = detector.predict(images)
                for (identifier, image), detections in zip(batch, predictions):
                    if output is not None:
                        draw_detections(image, detections, detector.class_names).save(output / f"frame{count:08d}.jpg")
                        records.write(json.dumps({"source": identifier, "detections": detections.tolist()}) + "\n")
                    count += 1
    finally:
        stream.close()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="ONNX file produced by task=export")
    parser.add_argument("--source", required=True, help="Image, image directory, video, webcam index or stream URL")
    parser.add_argument("--output", default="runs/onnx", help="Directory for JPEGs and predictions.jsonl")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument(
        "--nms-free", action="store_true", help="Fallback for one-to-one exports without postprocessing metadata"
    )
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    parser.add_argument("--threads", type=int, default=0, help="ONNX intra-op threads; 0 uses runtime default")
    parser.add_argument("--class-num", type=int, help="Required for old exports without metadata")
    parser.add_argument("--output-format", choices=["auto", "dfl", "xyxy"], default="auto")
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--strides", type=int, nargs="+", default=[8, 16, 32])
    args = parser.parse_args()
    detector = ONNXDetector(
        args.model,
        providers=args.providers,
        class_num=args.class_num,
        output_format=args.output_format,
        reg_max=args.reg_max,
        strides=args.strides,
        confidence=args.confidence,
        iou_threshold=args.iou,
        max_detections=args.max_detections,
        threads=args.threads,
        nms_free=args.nms_free,
    )
    count = run_inference(detector, args.source, args.output)
    print(f"Processed {count} frames; saved results to {args.output}")


if __name__ == "__main__":
    main()
