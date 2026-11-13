import math

import torch
from torch import nn

from torch_dnn.modules import (
    SemanticHead, FeaturePyramidNetwork, Location2D, BackboneNetwork,
    KERAS_BN_EPS, KERAS_BN_MOMENTUM, init_keras_style, level_number, set_batchnorm,
)

# Architecture version, stored in every checkpoint as the ``arch_version`` buffer, so
# that weights are only ever loaded into the network they were trained with.
#   1: the original torch_dnn network (before November 2026). Its checkpoints, such as
#      ~/.deepcell/models/dnn/saved_model_best_dict.pth, have no ``arch_version`` key.
#      Its feature pyramid passed the smoothed output down, built P6 from P5 and had no
#      ReLU before P7, it used PyTorch's BatchNorm settings, and its semantic heads fused
#      every pyramid level. This code does not build it, and those weights cannot be
#      loaded.
#   2: not used by torch_dnn (in torch_mesmer, an unreleased intermediate network).
#   3: the current network, the same design as torch_mesmer's version 3. The feature
#      pyramid is wired as in deepcell, BatchNorm settings and initialisation follow
#      Keras, and each semantic head reads only the finest pyramid level, as in
#      deepcell's PanopticNet. Pyramid convs that nothing reads are not built.
ARCH_VERSION = 3
LEGACY_ARCH_VERSION = 1

# The DynamicNuclearNet network, as deepcell-tf builds it in
# notebooks/training/panopticnets/"Nuclear Segmentation - DeepWatershed.ipynb":
# PanopticNet(backbone='efficientnetv2bl', location=True, input_shape=(256, 256, 1),
# backbone_levels=C1-C5, pyramid_levels=P1-P7, num_semantic_classes=[1, 1, 2]).
# ``use_imagenet`` is left at its default (True), as in the notebook; the app passes
# ``use_imagenet=False`` because it overwrites every weight with a checkpoint anyway.
DNN_MODEL_KWARGS = {
    'backbone': 'efficientnetv2bl',
    'crop_size': 256,
    'in_channels': 1,
    'backbone_levels': ['C1', 'C2', 'C3', 'C4', 'C5'],
    'pyramid_levels': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'],
    'n_semantic_classes': [1, 1, 2],  # inner distance, outer distance, fgbg
}

# The network's output channels (see ``PanopticNet``).
DNN_OUTPUT_CHANNELS = sum(DNN_MODEL_KWARGS['n_semantic_classes'])


def checkpoint_arch_version(state_dict):
    """Architecture version a state dict was saved from.

    Args:
        state_dict (dict): a ``PanopticNet`` state dict, e.g. from ``torch.load``.

    Returns:
        int: its ``arch_version`` entry, or ``LEGACY_ARCH_VERSION`` (1) if it has none,
        as for checkpoints of the original torch_dnn network.
    """
    version = state_dict.get('arch_version')
    return LEGACY_ARCH_VERSION if version is None else int(version)


class PanopticNet(nn.Module):
    '''
    DynamicNuclearNet's network, after deepcell's PanopticNet: backbone, feature
    pyramid and semantic heads. The defaults build DynamicNuclearNet
    (``DNN_MODEL_KWARGS``).

    Two coordinate channels (``Location2D``) are added to the image, and a 1x1 conv
    maps the result to the 3 channels the backbone expects. The backbone feeds a
    feature pyramid, and one semantic head per entry of ``n_semantic_classes``
    reads its finest level and returns a map at the input's resolution.

    For DynamicNuclearNet (``n_semantic_classes=[1, 1, 2]``) there are three
    semantic heads, in deepcell's order. Their outputs are concatenated into 4
    channels:

        0     inner distance (ReLU): how far inside a nucleus a pixel is
        1     outer distance (ReLU): distance to the nucleus edge
        2-3   fgbg (softmax), in deepcell's one-hot order: background, foreground

    ``postprocess_utils.DNN_HEAD_INDICES`` records which channels the watershed
    uses: as in deepcell's NuclearSegmentation, the seeds come from the inner
    distance and the mask from the outer distance. The fgbg head is only trained.

    Args:
        backbone (str): ``'efficientnetv2bl'`` (DynamicNuclearNet) or ``'resnet50'``;
            see ``BackboneNetwork``.
        feature_size (int): channels of every pyramid level.
        crop_size (int): side of the square input; must be a power of 2 and at least
            ``2 ** (coarsest pyramid level)``.
        use_imagenet (bool): start the backbone from ImageNet weights (downloaded on first use).
        backbone_levels (list[str]): the backbone levels the pyramid is built from.
        pyramid_levels (list[str]): the FPN levels. As in deepcell's PanopticNet, the heads
            read only the finest one (P1 here), so the coarser levels' output convs are
            not built. Their number still sets the smallest allowed input,
            ``2 ** (coarsest level)``.
        interpolation (str): ``'bilinear'`` or ``'nearest'``, for every 2x upsampling.
        n_semantic_classes (list[int]): output channels of each head, in order. A 1-channel
            head ends in ReLU, a head with more channels in softmax.
        in_channels (int): number of image channels (1 for DynamicNuclearNet: nuclear).
    '''

    def __init__(
            self,
            backbone=DNN_MODEL_KWARGS['backbone'],
            feature_size=256,
            crop_size=DNN_MODEL_KWARGS['crop_size'],
            use_imagenet=True,
            backbone_levels=DNN_MODEL_KWARGS['backbone_levels'],
            pyramid_levels=DNN_MODEL_KWARGS['pyramid_levels'],
            interpolation='bilinear',
            n_semantic_classes=DNN_MODEL_KWARGS['n_semantic_classes'],
            in_channels=DNN_MODEL_KWARGS['in_channels'],
    ):
        super().__init__()

        if interpolation not in ('bilinear', 'nearest'):
            raise ValueError(f"interpolation must be 'bilinear' or 'nearest', got {interpolation!r}")

        # Store configuration (copies, so the shared default lists are never changed)
        self.backbone = backbone
        self.use_imagenet = use_imagenet
        self.backbone_levels = list(backbone_levels)
        self.pyramid_levels = list(pyramid_levels)
        self.feature_size = feature_size
        self.interpolation = interpolation
        self.crop_size = crop_size
        self.n_semantic_classes = list(n_semantic_classes)
        self.in_channels = in_channels
        self.min_size = 2 ** max(level_number(p) for p in self.pyramid_levels)
        self._check_size(crop_size, crop_size, what='crop_size')

        # Build model
        self.location = Location2D()

        self.channel_conv = nn.Conv2d(
            in_channels=in_channels + 2,  # image + 2 location channels
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.bbnetwork = BackboneNetwork(
            backbone=self.backbone,
            use_imagenet=self.use_imagenet,
            keras_batchnorm=True,
        )

        # deepcell: target_level = min(semantic_levels); the heads read that level only.
        head_level = min(self.pyramid_levels, key=level_number)

        self.fpn = FeaturePyramidNetwork(
            levels=self.pyramid_levels,
            feature_size=self.feature_size,
            interpolation=self.interpolation,
            backbone_levels=self.backbone_levels,
            backbone_channels=self.bbnetwork.out_channels,
            output_levels=[head_level],
        )

        self.semantic_heads = nn.ModuleList([
            SemanticHead(
                n_classes=i,
                level=head_level,
                feature_size=self.feature_size,
                n_upsample=level_number(head_level),
                interpolation=self.interpolation,
            )
            for i in self.n_semantic_classes
        ])

        self._init_keras_style()
        # Saved with the weights, so a checkpoint records which network it belongs to.
        self.register_buffer('arch_version', torch.tensor(ARCH_VERSION))

    def _init_keras_style(self):
        """Give every layer outside the backbone Keras's initial weights and BatchNorm
        settings, as in deepcell's TF models. The backbone keeps its ImageNet weights,
        or torchvision's own init without them. Draws from the global random generator."""
        for part in (self.channel_conv, self.fpn, self.semantic_heads):
            init_keras_style(part)
        set_batchnorm(self.semantic_heads, eps=KERAS_BN_EPS, momentum=KERAS_BN_MOMENTUM)

    @property
    def version(self):
        """Architecture version of this network (``ARCH_VERSION``)."""
        return ARCH_VERSION

    def _check_size(self, h, w, what='input'):
        # deepcell's PanopticNet requires square, power-of-2 inputs (panopticnet.py:170-182).
        # They must also be at least 2**(coarsest level), so every pyramid level is >= 1 pixel.
        if h != w or h < self.min_size or not math.log2(h).is_integer():
            raise ValueError(
                f'{what} must be square with a power-of-2 side of at least {self.min_size}, '
                f'got {h}x{w}')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """``nn.Module.load_state_dict``, with a clear ``ValueError`` when the checkpoint
        is for another architecture version (such as the original torch_dnn network) or
        does not fit this network's layers (such as a Mesmer checkpoint)."""
        found = checkpoint_arch_version(state_dict)
        if found != self.version:
            if found == LEGACY_ARCH_VERSION:
                hint = ('it has no arch_version entry, so it was saved from the original '
                        'torch_dnn network (before November 2026), which this code no longer '
                        'builds. Those weights cannot be used; train a new model with '
                        '`python -m torch_dnn.train`')
            else:
                hint = 'this code does not build that version (see torch_dnn.model.ARCH_VERSION)'
            raise ValueError(
                f'Checkpoint is architecture version {found} but this PanopticNet is version '
                f'{self.version}: {hint}.')
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError as e:
            raise ValueError(
                'Checkpoint does not fit this network: its layer names or shapes differ '
                '(for example a Mesmer checkpoint, or one trained with other model settings '
                f'than torch_dnn.model.DNN_MODEL_KWARGS). Details: {e}') from e

    def forward(self, x, format='channels_first'):
        """Run the network on a batch of images.

        Args:
            x (torch.Tensor): float images, ``(B, in_channels, H, W)``, or
                ``(B, H, W, in_channels)`` with ``format='channels_last'``. They must be
                square, with a power-of-2 side of at least ``min_size``.
            format (str): the axis order of ``x``, ``'channels_first'`` or ``'channels_last'``.

        Returns:
            torch.Tensor: ``(B, sum(n_semantic_classes), H, W)``, always channels first:
            the heads' outputs, concatenated in order. For DynamicNuclearNet that is the
            4-channel layout in the class docstring.

        Raises:
            ValueError: for an unknown ``format``, or an input of the wrong shape or size.
        """
        if format == 'channels_last':
            x = x.permute(0, 3, 1, 2)
        elif format != 'channels_first':
            raise ValueError(f"format must be 'channels_first' or 'channels_last', got {format!r}")

        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f'expected input of shape (batch, {self.in_channels}, H, W) '
                             f'(channels first), got {tuple(x.shape)}')
        self._check_size(x.shape[2], x.shape[3])

        location = self.location(x)
        x = torch.cat([x, location], dim=1)
        x = self.channel_conv(x)

        backbone_features = self.bbnetwork(x)
        pyramid_features = self.fpn(backbone_features)

        predictions = []
        for head in self.semantic_heads:
            predictions.append(head(pyramid_features))

        return torch.cat(predictions, dim=1)
