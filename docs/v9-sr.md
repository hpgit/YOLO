# YOLOv9 SR variants

Select `model=v9-sr-t`, `model=v9-sr-s`, `model=v9-sr-m`, or `model=v9-sr-c`.
For training from scratch, set `weight=null`, for example:

```sh
yolo task=train model=v9-sr-t weight=null
```

These configurations preserve the corresponding v9 architecture and Main/AUX
heads. The t/s/c variants preserve the original channels; m widens selected
channels as described below. The t/s/m variants replace AConv with AConv2; c
replaces ADown with ADown2. `activation: Hardswish` applies to every active Conv/RepConv
activation, including nested blocks and both detection heads. Linear RepConv
branches and detection output layers remain linear. Existing v9 configurations
retain their original activations and pooling.

`v9-sr-m` aligns every internal convolution's total input channel count to a
multiple of 32, including nested CSP bottlenecks, detection heads, fixed
smoothing convolutions, and the AUX branch. RGB stems retain 3 input channels,
and the fixed DFL Conv3d retains `reg_max=16`. Channel counts never decrease:

- Feature widths 240, 360, and 184 become 256, 384, and 192, respectively.
- RepNCSPELAN blocks with 480 output channels keep that output width, but their
  `part_channels` increases from 480 to 512 so both nested channel splits align.
- AUX projection and fusion widths follow the corresponding feature widths.

Alignment refers to the total input tensor channels, not channels per group in
grouped/depthwise convolutions. Class logits and regression outputs retain their
original dimensions. Earlier `v9-sr-m` checkpoints have incompatible shapes for
the widened layers; train the revised configuration with `weight=null`.

AConv2/ADown2 replace the initial average pooling with a depthwise convolution
using the same kernel independently for each input channel:

```text
  5  27   5
 27 127  27    / 255
  5  27   5
```

This smoothing convolution has stride 1, zero padding 1, no bias, no batch
normalization, and no activation. The subsequent stride-2 convolutions and
ADown's max-pooling branch are preserved. Output dimensions are ceil(H/2) by
ceil(W/2); for even inputs they match the original blocks. For odd inputs the
original blocks use floor(H/2) by floor(W/2). Use input dimensions divisible by
32 for the complete models' feature concatenations.

The kernel is a non-persistent buffer, never an optimizer parameter. It remains
fixed even after `model.requires_grad_(True)`, follows device/dtype conversions,
and permits gradients through the input. It is omitted from state dictionaries
so EMA and checkpoint loading cannot change it; constructing the module
recreates the fixed kernel before restoring a checkpoint.

Tests cover kernel values, channel isolation, padding, backward propagation,
SGD/AdamW updates, reloads, EMA, and all four models' Main/AUX output shapes.
The m variant also checks actual convolution input widths and verifies that no
convolution input/output width shrinks relative to v9-m.
These checks establish execution correctness; detection accuracy is unmeasured.
