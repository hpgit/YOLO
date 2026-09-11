Custom Task
===========

Detection export
----------------

``task=export`` runs directly without constructing a Lightning Trainer, downloading
a dataset, or starting WandB. It loads weights using the existing model loader;
``weight=False`` explicitly exports an untrained model for conversion tests.

Install ONNX dependencies with ``pip install -e '.[export-onnx]'``. For TFLite,
use a separate Python 3.11+ environment and install
``pip install -e '.[export-tflite]'``. The latter uses Google's
`LiteRT Torch converter <https://github.com/google-ai-edge/litert-torch/blob/main/docs/pytorch_converter/README.md>`_
and may require a newer PyTorch than the training environment.

Example::

    python yolo/lazy.py task=export task.format=onnx model=v9-t weight=weights/v9-t.pt
    python yolo/lazy.py task=export task.format=tflite model=v9-t weight=weights/v9-t.pt
    python yolo/lazy.py task=export task.format=onnx image_size=[640,384] task.batch_size=2

The input is float32 RGB NCHW, normalized to [0, 1]. The ``image_size`` setting is
``[width, height]``, with each dimension a positive multiple of 32. Both formats
keep spatial dimensions fixed. ONNX supports an optional dynamic batch axis via
``task.dynamic_batch=true``; TFLite uses ``task.batch_size`` as a fixed dimension.
The default ONNX opset is 17 (configurable with ``task.opset``).

The single float32 output is ``[B, N, 4 + C]``. Boxes from all Main detection
scales are concatenated in head order. Each row is
``[x1, y1, x2, y2, class_0_score, ..., class_C-1_score]``. Coordinates are decoded
in input-image pixels, and may extend outside the image. Confidence is sigmoid
class probability for YOLOv9 and sigmoid class probability multiplied by sigmoid
objectness for YOLOv7; there is no separate objectness column. YOLOv9 rows are in
spatial row-major order per scale; YOLOv7 uses anchor then spatial row-major order.

For the supplied stride-8/16/32 YOLOv9 detection models at 640 by 640, N is 8400;
YOLOv7 uses three anchors at each location, giving N = 25200. No NMS, confidence
thresholding, top-k selection, coordinate clipping, inverse letterbox transform,
or auxiliary prediction output is included. Classification and segmentation
models are not supported by this detection export task.

Export replaces DFL's 5-D Conv3d calculation in a copy of the model with an
equivalent 4-D Conv2d calculation. Spatial dimensions are flattened before
softmax, and the checkpoint's projection weights are preserved. Training,
ordinary inference, and checkpoint formats are unchanged. Internal ONNX tensors,
including intermediate results, constants, and weights, must have rank at most 4.
Export infers and saves internal shapes and rejects unknown ranks or ranks above 4;
this also applies when the batch dimension is dynamic. Re-export older files to
apply this change.

The default path is ``${out_path}/export/${name}/${model.name}.${task.format}``.
Set ``task.output=path/model.onnx`` or ``task.output=path/model.tflite`` to override
it, and ``exist_ok=false`` to prevent overwriting. ONNX graph inputs and outputs
are named ``images`` and ``predictions``; TFLite tensor names are converter-defined,
so obtain their indices from the runtime's input/output details.

This task's decoded ONNX output is intended for external deployment. The existing
``task=inference task.fast_inference=onnx`` loader uses its own raw-head ONNX format
and is not a reader for these exported files.

Run the runtime equivalence checks with::

    python -m pytest tests/test_tools/test_export.py -q

The ONNX and TFLite integration tests require their corresponding optional
dependencies; unavailable runtimes are reported as skipped tests.
