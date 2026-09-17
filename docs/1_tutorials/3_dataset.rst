Create Dataset
==============

In this section, we will prepare the dataset and create a dataloader.

Overall, the dataloader can be created by:

.. code-block:: python

   from yolo import create_dataloader
   dataloader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task, use_ddp)

For inference, the dataset will be handled by :class:`~yolo.tools.data_loader.StreamDataLoader`, while for training and validation, it will be handled by :class:`~yolo.tools.data_loader.YoloDataLoader`.

The input arguments are:

- **DataConfig**: :class:`~yolo.config.config.DataConfig`, the relevant configuration for the dataloader.
- **DatasetConfig**: :class:`~yolo.config.config.DatasetConfig`, the relevant configuration for the dataset.
- **task_name**: :guilabel:`str`, the task name, which can be `inference`, `validation`, or `train`.
- **use_ddp**: :guilabel:`bool`, whether to use DDP (Distributed Data Parallel). Default is `False`.

Train and Validation
----------------------------

Multiple dataset inputs
~~~~~~~~~~~~~~~~~~~~~~~

``train`` and ``validation`` accept either one split name or a non-empty list
of split names under the shared ``path``:

.. code-block:: yaml

   path: data/custom
   train: [train_a, train_b]
   validation: [val_a, val_b]
   class_num: 2
   class_list: [person, dog]
   auto_download: null

Each entry uses the existing split layout: ``<path>/<split>.txt`` (an image
list, with paths relative to ``path`` or absolute), or
``<path>/images/<split>`` with ``<path>/labels/<split>`` or
``<path>/annotations/instances_<split>.json``. The entries themselves are
split names or explicit Parquet annotation paths, not arbitrary
image-directory or TXT-file paths.

Inputs are concatenated in configured order before training shuffle. All
inputs must share the same class numbering. Repeated inputs/images are kept;
avoid overlapping validation splits. Both legacy and YOLOv9 augmentation
sample from the combined dataset. Legacy rectangular batches are sorted by
aspect ratio across all inputs, and legacy caches remain separate per split.
Existing single-string configurations continue to work.

With ``evaluator: auto``, validation uses a combined COCO evaluator when every
input uses COCO JSON. Image and annotation IDs are remapped in memory, and
category IDs and names must match across JSON files. If any input uses TXT or Parquet,
the combined loader targets are evaluated with TorchMetrics. Metrics cover
the combined dataset; they are not averages of per-split AP values. An explicit
``annotation_path`` still selects one authoritative JSON; for multiple inputs,
its filenames should include the split directory relative to ``<path>/images``
or be absolute.

Parquet pseudo-label annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pandas DataFrames saved with ``df.to_parquet(...)`` are supported by both the
legacy and YOLOv9 loaders. Install the updated ``requirements.txt``, which
includes ``pandas`` and ``pyarrow``. Each row describes one box using these
columns (column order and the pandas index do not matter):

.. code-block:: text

   image, conf, id_class, box_cx, box_cy, box_w, box_h

``id_class`` is the zero-based index in the configured ``class_list``.
No category-ID remapping is performed:

.. code-block:: yaml

   path: data/custom
   train: [annotations/instances_train_a.parquet, train_b]
   validation: annotations/instances_val.parquet
   class_num: 2
   class_list: [person, dog]
   auto_download: null

Here ID 0 means person and ID 1 means dog. IDs must be integers in
``[0, class_num - 1]``; unknown or fractional indices are errors, not silently
discarded rows. Every Parquet input uses the same model class order. Ensure
that TXT and COCO JSON inputs have the same resulting order when mixing formats.

A ``dataset=parquet`` example config is included. All values can be overridden
through the CLI:

.. code-block:: bash

   .venv/bin/python yolo/lazy.py task=train dataset=parquet \
     dataset.path=/data/custom \
     'dataset.train=[annotations/instances_train_a.parquet,train_b]' \
     dataset.validation=annotations/instances_val.parquet \
     dataset.class_num=2 'dataset.class_list=[person,dog]'

``image`` is an image path. Rows for the same resolved image path are grouped
into one sample, retaining every box. ``conf`` is ignored: no threshold or
loss weighting is applied. Box centers and sizes must be finite and normalized
to [0, 1], with positive width and height. Derived corners may extend outside
the image and follow the existing TXT bbox clipping rules. No polygons are
invented for Copy-Paste.

Parquet files can be supplied as absolute paths or paths relative to
``dataset.path`` in ``train`` and ``validation``, including lists mixed with
existing split names. With a split name such as ``train_a``, the conventional
file is ``annotations/instances_train_a.parquet``. Existing ``train_a.txt``
and ``annotations/instances_train_a.json`` take precedence over automatic
Parquet discovery; an explicit ``.parquet`` path always selects that file.

Image lookup follows the COCO JSON loader. Absolute image paths are used
directly; relative image paths are tried in this order:

#. ``<dataset.path>/images/<split>/<image>``
#. ``<dataset.path>/images/<image>``
#. ``<dataset.path>/<image>``

For an explicit Parquet file, ``instances_train_a.parquet`` implies split
``train_a``; ``train_a.parquet`` does too. This also permits COCO-style bare
filenames as well as split-prefixed and dataset-relative image paths.

Only images represented by rows are included; an image with no rows is not
automatically added as a background sample. Missing files, missing columns,
invalid classes/boxes, and empty Parquet files produce errors. Parquet inputs
are read on each dataset construction, bypassing legacy ``.pache`` caches so
updated pseudo-labels are used immediately. Validation defaults to TorchMetrics
on the loaded targets. An explicit ``evaluator: coco`` requires a separate
COCO ``annotation_path`` when Parquet inputs are selected.

Dataloader Return Type
~~~~~~~~~~~~~~~~~~~~~

For each iteration, the return type includes:

- **batch_size**: the size of each batch, used to calculate batch average loss.
- **images**: the input images.
- **targets**: the ground truth of the images according to the task.

Auto Download Dataset
~~~~~~~~~~~~~~~~~~~~~

The dataset will be auto-downloaded if the user provides the `auto_download` configuration. For example, if the configuration is as follows:


.. literalinclude:: ../../yolo/config/dataset/mock.yaml
  :language: YAML


First, it will download and unzip the dataset from `{prefix}/{postfix}`, and verify that the dataset has `{file_num}` files.

Once the dataset is verified, it will generate `{train, validation}.cache` in Tensor format, which accelerates the dataset preparation speed.

Inference
-----------------

In streaming mode, the model will infer the most recent frame and draw the bounding boxes by default, given the save flag to save the image. In other modes, it will save the predictions to `runs/inference/{exp_name}/outputs/` by default.

Dataloader Return Type
~~~~~~~~~~~~~~~~~~~~~

For each iteration, the return type of `StreamDataLoader` includes:

- **images**: tensor, the size of each batch, used to calculate batch average loss.
- **rev_tensor**: tensor, reverse tensor for reverting the bounding boxes and images to the input shape.
- **origin_frame**: tensor, the original input image.

Input Type
~~~~~~~~~~

- **Stream Input**:

  - **webcam**: :guilabel:`int`, ID of the webcam, for example, 0, 1.
  - **rtmp**: :guilabel:`str`, RTMP address.

- **Single Source**:

  - **image**: :guilabel:`Path`, path to image files (`jpeg`, `jpg`, `png`, `tiff`).
  - **video**: :guilabel:`Path`, path to video files (`mp4`).

- **Folder**:

  - **folder of images**: :guilabel:`Path`, the relative or absolute path to the folder containing images.
