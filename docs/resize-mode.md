# Resize mode

`resize_mode=letterbox` (default) preserves the original aspect ratio and pads
with RGB `(0, 0, 0)`. `resize_mode=stretch` directly resizes to
`image_size=[width,height]`, scaling the horizontal and vertical axes independently.
Letterbox padding, mosaic canvases, and affine/perspective borders use black
RGB `(0, 0, 0)` across training, validation, inference, and ONNX inference.
Existing checkpoints remain loadable; the padding color changes their inputs.

Use the same mode for training, validation, inference and export:

```sh
yolo task=train resize_mode=stretch
yolo task=validation weight=weights/best.pt resize_mode=stretch
yolo task=inference weight=weights/best.pt resize_mode=stretch
yolo task=export weight=weights/best.pt task.format=onnx resize_mode=stretch
yolo task=inference weight=model.onnx resize_mode=stretch
```

The shared setting also propagates to training's nested validation configuration.
Each task exposes `task.data.resize_mode`; training's validation exposes
`task.validation.data.resize_mode`. Override these only for deliberate experiments.
The existing YOLOv9 training recipe still requires a fixed square input; the
legacy loader, validation and inference also support rectangular inputs.
Legacy `dynamic_shape` continues to select a size per batch before resizing.

For YOLOv9 training, stretch applies both to individual images and to each
source image before Mosaic composition. Box and polygon coordinates use their
respective horizontal and vertical scales. Mosaic, perspective and translation
augmentations can still introduce empty regions; this option removes aspect-ratio
padding during resizing, not the empty regions inherent in those augmentations.

Normalized bounding boxes stay unchanged under stretch. Inference restores pixel
coordinates using separate axis scales. The COCO evaluator uses the same policy
when mapping detections back to the annotation image dimensions.

ONNX export records the mode in `yolo.inference` metadata; preprocessing stays
outside the model graph. The portable script reads this metadata automatically:

```sh
python yolo/tools/onnx_inference.py --model model.onnx --source image.jpg
# Explicit override:
python yolo/tools/onnx_inference.py --model model.onnx --source image.jpg --resize-mode stretch
```

Old ONNX files without a resize policy default to letterbox. The Hydra entry
point uses its configured `resize_mode` (default letterbox), so pass
`resize_mode=stretch` there for stretch-trained/exported models. PyTorch weight
files do not automatically select the resize policy; keep the training setting
when validating or exporting weights.

Changing the mode on existing weights is supported, but its accuracy impact
requires evaluation on the intended dataset. Geometry and smoke tests establish
correct execution, not an AP improvement.
