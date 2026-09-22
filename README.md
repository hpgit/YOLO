# YOLO: Official Implementation of YOLOv9, YOLOv7, YOLO-RD

[![Documentation Status](https://readthedocs.org/projects/yolo-docs/badge/?version=latest)](https://yolo-docs.readthedocs.io/en/latest/?badge=latest)
![GitHub License](https://img.shields.io/github/license/WongKinYiu/YOLO)

[![Developer Mode Build & Test](https://github.com/WongKinYiu/YOLO/actions/workflows/develop.yaml/badge.svg)](https://github.com/WongKinYiu/YOLO/actions/workflows/develop.yaml)
[![Deploy Mode Validation & Inference](https://github.com/WongKinYiu/YOLO/actions/workflows/deploy.yaml/badge.svg)](https://github.com/WongKinYiu/YOLO/actions/workflows/deploy.yaml)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-green)](https://huggingface.co/spaces/henry000/YOLO)

<!-- > [!IMPORTANT]
> This project is currently a Work In Progress and may undergo significant changes. It is not recommended for use in production environments until further notice. Please check back regularly for updates.
>
> Use of this code is at your own risk and discretion. It is advisable to consult with the project owner before deploying or integrating into any critical systems. -->

Welcome to the official implementation of YOLOv7[^1] and YOLOv9[^2], YOLO-RD[^3]. This repository will contains the complete codebase, pre-trained models, and detailed instructions for training and deploying YOLOv9.

## TL;DR

- This is the official YOLO model implementation with an MIT License.
- For quick deployment: you can directly install by pip+git:

```shell
pip install git+https://github.com/WongKinYiu/YOLO.git
yolo task.data.source=0 # source could be a single file, video, image folder, webcam ID
```

## Introduction

- [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)
- [**YOLO-RD**: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](https://arxiv.org/abs/2410.15346)

## Installation

To get started using YOLOv9's developer mode, we recommand you clone this repository and install the required dependencies:

```shell
git clone git@github.com:WongKinYiu/YOLO.git
cd YOLO
pip install -r requirements.txt
```

## Features

<table>
<tr><td>

## Task

These are simple examples. For more customization details, please refer to [Notebooks](examples) and lower-level modifications **[HOWTO](docs/HOWTO.md)**.

## Training

For optional one-to-one training and inference without NMS, see the [NMS-free guide (한국어)](docs/nms-free.md).

For a reproducible COCO 1/20 subset and training/validation speed comparisons, see the [subset performance guide (한국어)](docs/performance-coco-subset.md).

Named runs automatically resume their latest checkpoint when `weight` is omitted or `weight=False`.
See [checkpoint saving and resumption (한국어)](docs/checkpoints.md) for filename formats, `best.pt`, and weight precedence.

To train YOLO on your machine/dataset:

1. Modify the configuration file `yolo/config/dataset/**.yaml` to point to your dataset.
2. Run the training script:

```shell
python yolo/lazy.py task=train dataset=** use_wandb=True
python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c weight=False # or more args
```

### Transfer Learning

To perform transfer learning with YOLOv9:

```shell
python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c dataset={dataset_config} device={cpu, mps, cuda}
```

With `+quiet=True`, each validated epoch prints a single line containing average losses,
AP/AR, and training/validation time and throughput. The same line is appended to
`result.log` in the experiment directory (by default, `runs/train/<name>/result.log`).
Resuming in the same directory preserves existing lines. Sanity validation is excluded,
and only the global-zero process writes the summary in distributed runs.

### Inference

To use a model for object detection, use:

```shell
python yolo/lazy.py # if cloned from GitHub
python yolo/lazy.py task=inference \ # default is inference
                    name=AnyNameYouWant \ # AnyNameYouWant
                    device=cpu \ # hardware cuda, cpu, mps
                    model=v9-s \ # model version: v9-c, m, s
                    task.nms.min_confidence=0.1 \ # nms config
                    task.data.source=data/toy/images/train \ # file, dir, webcam
                    +quiet=True \ # Quiet Output
yolo task.data.source={Any Source} # if pip installed
yolo task=inference task.data.source={Any}
```

For an exported ONNX model (including QDQ), use its file as `weight`:

```shell
pip install -e '.[export-onnx]'
python yolo/lazy.py task=inference weight=runs/export/v9-dev/v9-c.onnx \
    task.data.source=demo/images/inference/image.png name=onnx-demo
```

This runs ONNX Runtime without constructing a PyTorch model or Lightning Trainer.
Input size and decoder settings come from the ONNX artifact. Results are saved as
`runs/inference/onnx-demo/frame00000000.jpg` and `predictions.jsonl`;
`task.save_predict=false` disables saving. CPU is the default; select runtime
providers explicitly with `task.onnx.providers=[CUDAExecutionProvider,CPUExecutionProvider]`
when using `onnxruntime-gpu`. `device` and `accelerator` do not select ONNX providers.

**Portable single-file inference:** copy [onnx_inference.py](yolo/tools/onnx_inference.py)
and the ONNX model anywhere. The file needs no repository installation or PyTorch:

```shell
pip install numpy Pillow onnxruntime
python onnx_inference.py --model model.onnx --source image.jpg --output results
# Image folders, videos and webcams are also supported (video/webcam needs OpenCV).
pip install opencv-python-headless
python onnx_inference.py --model model.onnx --source video.mp4 --output results
```

The file includes RGB letterboxing, DFL decoding, class-aware multi-label NMS,
original-image coordinate restoration, drawing, and JSONL output. See
[ONNX inference details](docs/4_deploy/2_onnx.rst) for the Python API, old export
compatibility and options.

### Validation

To validate model performance, or generate a json file in COCO format:

```shell
python yolo/lazy.py task=validation
python yolo/lazy.py task=validation dataset=toy
```

### Export (ONNX / TFLite)

```shell
pip install -e '.[export-onnx]'
python yolo/lazy.py task=export task.format=onnx model=v9-c weight=weights/v9-c.pt
python yolo/lazy.py task=export task.format=onnx task.dynamic_batch=true task.output=runs/model.onnx

# TFLite: use a separate Python >= 3.11 environment for the converter dependencies.
pip install -e '.[export-tflite]'
python yolo/lazy.py task=export task.format=tflite model=v9-c weight=weights/v9-c.pt
```

YOLOv9 ONNX has one float32 output: `[batch_size, num_candidates, 4 * reg_max + num_classes]`.
Each row contains four Softmax distributions first, then sigmoid class probabilities:
`[left_bins..., top_bins..., right_bins..., bottom_bins..., class_0, ..., class_C-1]`.
Softmax normalizes each direction's `reg_max` bins independently. The application
computes DFL expectation and decodes coordinates; this output contains no decoded boxes.
This replaces the previous YOLOv9 ONNX `[B,N,4+C]` contract, so update consumers when re-exporting.

TFLite and YOLOv7 ONNX retain decoded `[B,N,4+C]` output:
`[x1, y1, x2, y2, class_0_score, ...]` in input-image pixels. YOLOv7 scores include objectness.
All exports exclude auxiliary outputs, NMS and confidence filtering. Internal ONNX
tensors, constants, and weights are checked to have at most four dimensions.
FP training and checkpoint formats are unchanged; ordinary export does not itself perform INT8 quantization.
For YOLOv9 convolution QAT and encoding-preserving QDQ ONNX, see [QAT guide](docs/qat.md).
Use `qat.enabled=true` for fine-tuning and `task.qdq=true` to export a trained QAT checkpoint.
Input is float32 RGB `[batch_size, 3, height, width]`, scaled to `[0, 1]`; perform
resize/letterbox preprocessing and NMS in your application.

`image_size=[640,640]` means `[width,height]` (positive multiples of 32), and
`task.batch_size=1` sets the export batch size. At 640×640 with 80 classes,
YOLOv9 ONNX with `reg_max=16` returns `[1,8400,144]`; TFLite returns `[1,8400,84]`.
Spatial dimensions are fixed for both formats;
`task.dynamic_batch=true` is ONNX-only. Files default to
`runs/export/<name>/<model>.onnx` or `.tflite`; use `task.output` to choose a path.
Use the same `model` and `dataset` configuration as the trained checkpoint.
See [export details](docs/3_custom/3_task.rst) for the output contract and dependencies.

## Contributing

Contributions to the YOLO project are welcome! See [CONTRIBUTING](docs/CONTRIBUTING.md) for guidelines on how to contribute.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MultimediaTechLab/YOLO&type=Date)](https://star-history.com/#MultimediaTechLab/YOLO&Date)

## Citations

```
@inproceedings{wang2022yolov7,
      title={{YOLOv7}: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors},
      author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
      year={2023},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},

}
@inproceedings{wang2024yolov9,
      title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
      author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
      year={2024},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
}
@inproceedings{tsui2024yolord,
      author={Tsui, Hao-Tang and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
      title={{YOLO-RD}: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary},
      booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2025},
}

```

[^1]: [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)

[^2]: [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)

[^3]: [**YOLO-RD**: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](https://arxiv.org/abs/2410.15346)
