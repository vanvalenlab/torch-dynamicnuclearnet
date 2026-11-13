"""Training and validation data for DynamicNuclearNet.

Copied from ``torch_mesmer.loaders`` and changed only where deepcell's
DynamicNuclearNet training notebook (``Nuclear Segmentation -
DeepWatershed.ipynb``) differs from the Mesmer recipe:

* one nuclear channel in, one label channel;
* targets ``inner-distance``, ``outer-distance``, ``fgbg`` with the notebook's
  keyword arguments;
* normalisation is ``histogram_normalization`` with its defaults (CLAHE with
  ``kernel_size=None``, i.e. 1/8 of the image side), on the full image, before
  augmentation; there is no percentile clip;
* zoom range ``(0.75, 1/0.75)``;
* the only quality control is deepcell's ``min_objects=1`` (images without any
  object are left out), applied to train *and* val as in the notebook; there is
  no small-object removal.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_dnn.transforms import transform_masks
from torch_dnn.augmentation import MultiTransform
from torch_dnn.utils import histogram_normalization

# Targets, normalisation and quality control of the DynamicNuclearNet notebook.
IN_TRANSFORMS = ('inner-distance', 'outer-distance', 'fgbg')
TRANSFORMS_KWARGS = {
    'outer-distance': {'erosion_width': 1},
    'inner-distance': {'alpha': 'auto', 'beta': 1, 'erosion_width': 0},
}
CLAHE_KERNEL_SIZE = None  # histogram_normalization default: 1/8 of the image side (64 px for 512 px images)
ZOOM_MIN = 0.75           # zoom range (0.75, 1/0.75)
MIN_OBJECT_SIZE = 0       # no small-object removal
MIN_OBJECTS = 1           # deepcell min_objects: drop images with fewer objects than this
LABEL_CHANNEL = 0         # the label channel the object count uses (there is only one)


def target_layout(in_transforms, transforms_kwargs=None):
    """Channels and interpolation mode of each target, for one compartment.

    Returns:
        list of ``(n_channels, mode)``, one per entry of ``in_transforms``.
        Continuous maps are resized bilinearly; one-hot maps with nearest.
    """
    transforms_kwargs = transforms_kwargs or {}
    layout = []
    for transform in in_transforms:
        key = transform.lower().replace('_', '-')
        if key in ('inner-distance', 'outer-distance'):
            layout.append((1, 'bilinear'))
        elif key == 'pixelwise':
            separate = transforms_kwargs.get(transform, {}).get('separate_edge_classes', False)
            layout.append((4 if separate else 3, 'nearest'))
        elif key == 'fgbg':
            layout.append((2, 'nearest'))
        else:
            raise ValueError(f'`{transform}` is not a valid transform')
    return layout


def remove_small_objects(labels, min_size):
    """Zero out every label with fewer than ``min_size`` pixels.

    Same as ``skimage.morphology.remove_small_objects`` on a label image: sizes
    are counted per label value, not per connected component.
    """
    labels = np.asarray(labels).astype(np.int64)
    if min_size <= 0 or not labels.any():
        return labels
    sizes = np.bincount(labels.ravel())
    too_small = sizes < min_size
    too_small[0] = False
    labels[too_small[labels]] = 0
    return labels


def count_objects(labels, min_size=0):
    """Number of objects in a label image.

    With ``min_size=0`` this is deepcell's count, ``len(np.unique(labels)) - 1``
    (``SemanticIterator`` in ``deepcell/image_generators/semantic.py``), which
    assumes the image has background. Otherwise, the number of labels with at
    least ``min_size`` pixels (background excluded).
    """
    labels = np.asarray(labels)
    if min_size <= 0:
        return int(len(np.unique(labels)) - 1)
    sizes = np.bincount(labels.astype(np.int64).ravel())[1:]
    return int(np.count_nonzero(sizes >= min_size))


def qc_keep_indices(y, min_objects=MIN_OBJECTS, min_object_size=MIN_OBJECT_SIZE, channel=LABEL_CHANNEL):
    """Indices of the images with at least ``min_objects`` objects.

    With the defaults this is deepcell's ``min_objects=1`` filter: images
    whose label image has no object are dropped. If ``min_object_size`` is set,
    objects smaller than it are not counted. Images are read one at a time, so
    a zarr array is never loaded whole.
    """
    keep = []
    for i in range(y.shape[0]):
        if count_objects(y[i, channel], min_object_size) >= min_objects:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


class SegmentationDataset(Dataset):
    """Images and their training targets, at native resolution.

    ``X`` and ``y`` are ``(N, C, H, W)`` arrays (NumPy or zarr; zarr is read one
    image at a time). Each item is normalised (``histogram_normalization``),
    its labels are turned into targets, and for ``dataset_type='train'`` one
    random augmentation is applied to the image and every target.

    Target channels are ordered compartment by compartment, then transform by
    transform: for the default ``in_transforms`` and one label channel that is
    ``[inner distance, outer distance, background, foreground]``, the model's
    output order.

    Args:
        X, y: images and integer label images.
        dataset_type: ``'train'`` (augmented) or ``'val'`` (centre crop only).
        in_transforms: target transforms applied to every label channel.
        transforms_kwargs: keyword arguments for each transform, keyed by name.
        crop_size: output side.
        rotation_range: augmentation rotation limit, degrees.
        zoom_min: augmentation zoom range is ``(zoom_min, 1/zoom_min)`` on each axis.
        kernel_size: CLAHE kernel size (``None``: 1/8 of the image side).
        min_object_size: objects smaller than this are removed from every label
            channel before the targets are computed, and are not counted by
            ``min_objects``. 0 disables it.
        min_objects: images with fewer objects than this in ``qc_channel`` are
            left out. 0 disables it.
        qc_channel: label channel ``min_objects`` counts in.
        semantic_heads: optional; if given, it must match the target channels
            implied by ``in_transforms`` (a guard against a model/data mismatch).
    """

    def __init__(self,
                 X,
                 y,
                 dataset_type='train',
                 in_transforms=IN_TRANSFORMS,
                 transforms_kwargs=None,
                 crop_size=256,
                 rotation_range=180,
                 zoom_min=ZOOM_MIN,
                 kernel_size=CLAHE_KERNEL_SIZE,
                 min_object_size=MIN_OBJECT_SIZE,
                 min_objects=0,
                 qc_channel=LABEL_CHANNEL,
                 semantic_heads=None,
                 ):

        if X.shape[0] != y.shape[0]:
            raise ValueError(f'X has {X.shape[0]} images but y has {y.shape[0]}')
        if X.ndim != 4 or y.ndim != 4:
            raise ValueError(f'X and y must be (N, C, H, W); got {X.shape} and {y.shape}')

        self.X = X
        self.y = y
        self.dataset_type = dataset_type
        self.in_transforms = list(in_transforms)
        self.transforms_kwargs = TRANSFORMS_KWARGS if transforms_kwargs is None else transforms_kwargs
        self.crop_size = crop_size
        self.rotation_range = rotation_range
        self.zoom_min = zoom_min
        self.kernel_size = kernel_size
        self.min_object_size = min_object_size

        # The augmentation sees one stacked tensor: the image channels, then for
        # each compartment the targets of each transform. It needs one
        # interpolation mode per channel, so each target's mode is repeated
        # once per channel of that target.
        layout = target_layout(self.in_transforms, self.transforms_kwargs) * y.shape[1]
        self.head_channels = [n for n, _ in layout]
        self.interpolation = ['bilinear'] * X.shape[1]
        for n_channels, mode in layout:
            self.interpolation += [mode] * n_channels

        if semantic_heads is not None and list(semantic_heads) != self.head_channels:
            raise ValueError(
                f'semantic_heads={list(semantic_heads)}, but {self.in_transforms} on {y.shape[1]} '
                f'label channels gives heads {self.head_channels}')

        if min_objects > 0:
            self.indices = qc_keep_indices(y, min_objects, min_object_size, qc_channel)
        else:
            self.indices = np.arange(X.shape[0])

    def _normalize(self, image):
        """CLAHE (``histogram_normalization``), channel by channel.

        Args:
            image: ``(C, H, W)``.

        Returns:
            ``(C, H, W)``, float32. The helper works on batches, so a batch
            axis is added for it and removed again.
        """
        image = np.expand_dims(image, 0)
        image = histogram_normalization(image, kernel_size=self.kernel_size)
        return image.squeeze(axis=0)

    def _transform_labels(self, y):
        """Turn one image's label channels into its training targets.

        Objects smaller than ``min_object_size`` are removed first.

        Args:
            y: integer labels, ``(compartments, H, W)``.

        Returns:
            ``(sum(head_channels), H, W)``: compartment by compartment, the
            target of each transform in ``in_transforms`` (see the class
            docstring for the order).
        """
        y_semantic = []

        for compartment in range(y.shape[0]):
            labels = y[compartment]
            if self.min_object_size > 0:
                labels = remove_small_objects(labels, self.min_object_size)
            labels = np.expand_dims(labels, 0)

            for transform in self.in_transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = transform_masks(labels, transform, unbatched=True, **transform_kwargs)
                y_semantic.append(y_transform.squeeze(axis=0))

        return np.concatenate(y_semantic, axis=0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Load, normalise, build the targets of and augment one sample.

        Returns:
            ``(image, targets)``, float32 tensors of ``(C_image, crop_size,
            crop_size)`` and ``(sum(head_channels), crop_size, crop_size)``.
        """
        i = int(self.indices[idx])
        x = np.asarray(self.X[i])
        y = np.asarray(self.y[i])

        x = self._normalize(x)
        y = self._transform_labels(y)

        # Stack x and y along the channel dimension, so one draw of the
        # augmentation moves them together.
        combined = torch.cat([torch.from_numpy(x).float(), torch.from_numpy(y).float()], dim=0)

        transform = MultiTransform(
            combined.shape[1:],
            dataset_type=self.dataset_type,
            crop_size=self.crop_size,
            rotation_range=self.rotation_range,
            zoom_min=self.zoom_min,
        )
        combined = transform(combined, self.interpolation)

        # Split back into x and y
        n_x = x.shape[0]
        return combined[:n_x], combined[n_x:]


def _subset(split, max_images):
    """``split`` limited to its first ``max_images`` images (``None``: all of it).

    Slicing a zarr array reads it, so the subset is read into memory; use it
    for small debugging runs only.
    """
    if split is None or max_images is None:
        return split
    return {'X': np.asarray(split['X'][:max_images]), 'y': np.asarray(split['y'][:max_images])}


def create_data_loaders(
    train,
    val,
    crop_size=256,
    zoom_min=ZOOM_MIN,
    rotation_range=180,
    batch_size=16,
    num_workers=4,
    in_transforms=IN_TRANSFORMS,
    transforms_kwargs=None,
    kernel_size=CLAHE_KERNEL_SIZE,
    min_object_size=MIN_OBJECT_SIZE,
    min_objects=MIN_OBJECTS,
    semantic_heads=None,
    seed=0,
    max_train_images=None,
    max_val_images=None,
):
    """Build the train and val loaders.

    ``min_objects`` (and ``min_object_size``) are applied to train and val, as
    the notebook passes ``min_objects=1`` to both of its generators. Both
    loaders drop the last partial batch, as the recipe does (it runs
    ``n // batch_size`` steps per epoch, for training and validation). Train is
    shuffled and augmented; val is neither.

    Args:
        train, val: mappings with ``'X'`` (images) and ``'y'`` (integer
            labels), both ``(N, C, H, W)``, channels first, e.g. an open zarr
            group. Either can be ``None``, and then so is its loader.
        crop_size, zoom_min, rotation_range, in_transforms, transforms_kwargs,
            kernel_size: passed to ``SegmentationDataset``.
        batch_size: samples per batch.
        num_workers: DataLoader worker processes.
        min_object_size, min_objects: quality control, see ``SegmentationDataset``.
        semantic_heads: optional check of the target channels against the
            model's heads, see ``SegmentationDataset``.
        seed: seeds the train shuffle order and, through it, each worker's
            augmentation draws (PyTorch seeds ``torch``, ``random`` and
            ``numpy`` in every worker from the loader's generator).
        max_train_images, max_val_images: use only the first this many images
            of each split (for quick tests; ``None`` uses all of them).

    Returns:
        ``(train_loader, val_loader)``. Each batch is ``(images, targets)``,
        float32 tensors of ``(batch_size, C_image, crop_size, crop_size)`` and
        ``(batch_size, sum(semantic_heads), crop_size, crop_size)``.
    """
    common = dict(
        in_transforms=in_transforms,
        transforms_kwargs=transforms_kwargs,
        crop_size=crop_size,
        rotation_range=rotation_range,
        zoom_min=zoom_min,
        kernel_size=kernel_size,
        min_object_size=min_object_size,
        min_objects=min_objects,
        semantic_heads=semantic_heads,
    )

    train = _subset(train, max_train_images)
    val = _subset(val, max_val_images)

    dataloader = None
    valloader = None

    if train is not None:

        train_dataset = SegmentationDataset(
            train['X'],
            train['y'],
            dataset_type='train',
            **common)

        generator = torch.Generator()
        generator.manual_seed(seed)

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=True, drop_last=True, generator=generator)

    if val is not None:
        val_dataset = SegmentationDataset(
            val['X'],
            val['y'],
            dataset_type='val',
            **common)

        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                               pin_memory=True, drop_last=True)

    return dataloader, valloader
