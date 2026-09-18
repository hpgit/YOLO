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

YOLOv9 ONNX output
~~~~~~~~~~~~~~~~~

The single float32 output is ``[B, N, 4*R + C]``, where ``R=reg_max`` (usually 16).
All Main detection scales are concatenated in head order, with spatial row-major
order within each scale. Each row is::

    [P_left(0), ..., P_left(R-1),
     P_top(0), ..., P_top(R-1),
     P_right(0), ..., P_right(R-1),
     P_bottom(0), ..., P_bottom(R-1),
     sigmoid(class_0), ..., sigmoid(class_C-1)]

Each direction's probabilities are produced by a separate Softmax over its R bins.
They sum to 1 independently; class probabilities use independent sigmoids and do
not need to sum to 1. The output contains probabilities, not logits or decoded boxes.
No DFL projection, stride multiplication or anchor-grid decoding is included.
Export remains float32; it does not insert INT8 quantization.

This is a breaking output-contract change from the previous YOLOv9 ONNX
``[B,N,4+C]`` decoded output. Update consumers when re-exporting. For postprocessing,
split the last C channels as class scores, reshape the first 4*R channels to
``[B,N,4,R]`` and compute ``distance = sum(probability[i] * i)`` for i=0..R-1.
Multiply L/T/R/B distances by each candidate's stride and decode with its anchor
center: ``xyxy = [anchor_x-left, anchor_y-top, anchor_x+right, anchor_y+bottom]``.
For the standard heads, anchor centers are ``(column+0.5, row+0.5) * stride``.
Use the model's actual strides and head order. Scores already have sigmoid applied,
and the box distributions already have Softmax applied; do not apply either again.

TFLite and YOLOv7 ONNX output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These paths retain the single float32 decoded output ``[B,N,4+C]`` with rows
``[x1,y1,x2,y2,class_0_score,...,class_C-1_score]``. Coordinates use input-image
pixels and may extend outside the image. YOLOv9 TFLite scores are sigmoid class
probabilities. YOLOv7 scores multiply sigmoid class probability by sigmoid
objectness, with no separate objectness column. YOLOv7 rows use anchor then spatial
row-major order within each scale.

Common export constraints
~~~~~~~~~~~~~~~~~~~~~~~~~

For the supplied stride-8/16/32 YOLOv9 detection models at 640 by 640, N is 8400;
YOLOv7 uses three anchors at each location, giving N = 25200. No NMS, confidence
thresholding, top-k selection, coordinate clipping, inverse letterbox transform,
or auxiliary prediction output is included. Classification and segmentation
models are not supported by this detection export task.

YOLOv9 ONNX replaces DFL in a copy of the model with a rank-4 probability path:
``[B,4*R,H,W] -> [B,4,R,H*W] -> [B,H*W,4,R] -> Softmax(last axis) -> [B,H*W,4*R]``.
The unused DFL projection is removed. Decoded export instead uses a rank-4 Conv2d
projection and preserves the checkpoint's projection weights. Training,
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

This task's ONNX output is intended for external deployment. The existing
``task=inference task.fast_inference=onnx`` loader uses its own raw-head ONNX format
and is not a reader for these exported files.

Run the runtime equivalence checks with::

    python -m pytest tests/test_tools/test_export.py -q

The ONNX and TFLite integration tests require their corresponding optional
dependencies; unavailable runtimes are reported as skipped tests.
