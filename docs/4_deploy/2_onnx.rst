.. _ONNX:

ONNX export and inference
=========================

Export and run in the repository
-------------------------------

.. code-block:: shell

   pip install -e '.[export-onnx]'
   python yolo/lazy.py task=export model=v9-c weight=weights/v9-c.pt task.output=runs/model.onnx
   python yolo/lazy.py task=inference weight=runs/model.onnx task.data.source=image.jpg name=onnx-demo

An ``.onnx`` weight automatically selects ONNX Runtime. Explicit
``task.fast_inference=onnx`` also selects it and requires an existing ONNX weight.
The runtime does not build a PyTorch model, download weights, start training
loggers, or initialize a Lightning Trainer. The model's input dimensions take
precedence over ``image_size``. Outputs go under
``<out_path>/inference/<name>/`` with the usual ``exist_ok`` directory behavior.
``task.save_predict=false`` processes the input without writing predictions.

Standalone single-file deployment
---------------------------------

Copy only ``yolo/tools/onnx_inference.py`` and your ONNX file to another directory
or computer. The script has no imports from this repository, PyTorch, torchvision,
Hydra, or Lightning. It runs directly without installing the ``yolo`` package:

.. code-block:: shell

   pip install numpy Pillow onnxruntime
   python onnx_inference.py --model model.onnx --source image.jpg --output results
   python onnx_inference.py --model model.onnx --source images/ --confidence 0.25 --iou 0.5

Images and recursive image directories use Pillow. For videos, webcam indices
such as ``--source 0``, and stream URLs, also install ``opencv-python-headless``.
Video processing consumes frames until EOF; streams run until EOF or Ctrl+C.
The script saves annotated JPEG frames rather than encoding an output video.
It does not open a display window. Fixed batches greater than one are supported:
the final incomplete batch repeats the last input, then discards padded results.
Dynamic-batch models also work; the command-line runner feeds them one frame at a time.

Each output directory contains ``frame00000000.jpg`` etc. and
``predictions.jsonl``. Each JSON line contains ``source`` and ``detections``;
each detection is ``[class_id, x1, y1, x2, y2, score]`` in original-image pixels.
An image with no surviving detections has an empty list. Video source identifiers
include a zero-based frame index. Reusing an output directory overwrites matching
frame names and the JSONL file; use a fresh directory for each standalone run.

Python API
----------

.. code-block:: python

   from PIL import Image
   from onnx_inference import ONNXDetector

   detector = ONNXDetector("model.onnx", confidence=0.25)
   with Image.open("image.jpg") as image:
       detections = detector.predict(image)[0]  # NumPy [N, 6], original coordinates

``predict`` also accepts a list of PIL images, returning a list of arrays.
For a fixed-batch model, provide exactly the exported batch size; for dynamic
batch, any positive size is accepted. ``detector(preprocessed_array)`` accepts
float32 RGB NCHW inputs scaled to [0, 1] and returns decoded ``[B,N,4+C]`` before
thresholding, clipping and NMS.

Output contracts and compatibility
----------------------------------

New exports embed ``yolo.inference`` JSON metadata with a version, output format,
class count and class names, plus DFL ``reg_max`` and strides in head order.
Metadata takes precedence over fallback arguments, so a custom model does not
need the original YAML files. Labels fall back to numeric IDs if the configured
class-name list does not match the exported class count.

YOLOv9 exports contain four L/T/R/B softmax distributions and sigmoid class
probabilities: ``[B,N,4*reg_max+C]``. Inference computes each distribution's
expectation over bins ``0..reg_max-1``, scales by its head stride and decodes around
the corresponding cell center. YOLOv7 and older decoded exports contain pixel
``xyxy`` boxes followed by class scores: ``[B,N,4+C]``. No second sigmoid,
softmax or objectness multiplication is applied. QDQ exports use the same float32
input/output contract and run through the same decoder.

For old exports without metadata, specify the original class count and any
non-default decoder settings. Auto detection compares the output width against
``4+C`` and ``4*reg_max+C``; it does not guess the class count:

.. code-block:: shell

   python onnx_inference.py --model old-v9.onnx --source image.jpg \
       --class-num 80 --reg-max 16 --strides 8 16 32
   python onnx_inference.py --model old-decoded.onnx --source image.jpg \
       --class-num 80 --output-format xyxy

Inside the repository, old files use ``dataset.class_num``,
``model.anchor.reg_max``, ``task.onnx.strides`` and ``task.onnx.output_format``
as fallbacks; select the same dataset/model settings used for export.
Legacy multi-output raw-head ONNX files from ``FastModelLoader`` are unsupported;
re-export with ``task=export``. Input spatial dimensions must be fixed and the
input/output tensors must be float32.

Postprocessing uses class-aware multi-label NMS on already-normalized scores,
limits results with ``--max-detections`` (default 300), reverses the actual rounded
letterbox resize, clips to the original image and drops zero-area boxes. A candidate
can yield detections for multiple classes. Preprocessing uses RGB, Pillow LANCZOS
resize and padding value 114, consistent with repository inference preprocessing.

Runtime providers
-----------------

CPUExecutionProvider is the explicit default. To use CUDA, install a compatible
``onnxruntime-gpu`` runtime and select it explicitly:

.. code-block:: shell

   python onnx_inference.py --model model.onnx --source image.jpg \
       --providers CUDAExecutionProvider CPUExecutionProvider
   python yolo/lazy.py task=inference weight=model.onnx task.data.source=image.jpg \
       'task.onnx.providers=[CUDAExecutionProvider,CPUExecutionProvider]'

Unavailable providers produce an error. ONNX providers are independent of the
repository's Lightning ``device``/``accelerator`` settings. NumPy postprocessing
always runs on CPU. ``--threads`` controls ONNX intra-op CPU threads (0 is the
runtime default); the repository adapter uses ``cpu_num``.
