"""Parity tests: the training data pipeline (``torch_dnn.loaders``) vs what
deepcell's DynamicNuclearNet notebook feeds the network.

The notebook calls ``histogram_normalization`` (deepcell-toolbox) on the whole
array, then ``CroppingDataGenerator.flow(..., min_objects=1, transforms=[...],
transforms_kwargs={...})``, whose iterator computes the targets of each full
image with ``_transform_masks`` before cropping and augmenting it. The
reference versions of those pieces come from ``tests/_reference.py``.

The port is channels-first ``(C, H, W)``; the reference is channels-last.
Augmentation itself is not compared here (it is random and implemented
differently on purpose; see ``torch_dnn/augmentation.py`` and
``tests/test_data_pipeline.py``). Inference-side parity (tiling, resizing,
``deep_watershed``) is tested with the inference code.
"""

import warnings

import numpy as np
import pytest
import scipy.ndimage as nd

from torch_dnn import utils as pu
from torch_dnn.loaders import (IN_TRANSFORMS, MIN_OBJECTS, TRANSFORMS_KWARGS, SegmentationDataset,
                               qc_keep_indices)

from tests.conftest import random_blobs


def cl(x):
    """channels-first -> channels-last"""
    return np.moveaxis(x, 1, -1)


def cf(x):
    """channels-last -> channels-first"""
    return np.moveaxis(x, -1, 1)


def _image(shape, seed=0, blank=None):
    """Positive, skewed "fluorescence" image, channels-last (B, H, W, C).
    ``blank=(b, c)`` zeros one channel to exercise the constant-channel path."""
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 1.0, size=shape).astype(np.float32)
    x = nd.gaussian_filter(x, sigma=(0, 1.5, 1.5, 0))
    if blank is not None:
        x[blank[0], ..., blank[1]] = 0
    return x


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kernel_size', [None, 64])
def test_histogram_normalization_matches(ref, kernel_size):
    x = _image((2, 96, 80, 1), blank=(1, 0))
    port = _quiet(pu.histogram_normalization, cf(x), kernel_size=kernel_size)
    refv = _quiet(ref.processing.histogram_normalization, x, kernel_size=kernel_size)
    np.testing.assert_array_equal(cl(port), refv)


def test_dataset_normalisation_is_the_notebooks(ref):
    """``histogram_normalization(X)`` with no kwargs, per full image, on uint16-valued data."""
    x = np.round(_image((1, 512, 512, 1), seed=3) * 3000).astype(np.uint16)
    ds = SegmentationDataset(cf(x).astype(np.float32), np.zeros((1, 1, 512, 512), np.int32))
    refv = _quiet(ref.processing.histogram_normalization, x)   # the notebook passes uint16
    np.testing.assert_array_equal(cl(ds._normalize(cf(x)[0].astype(np.float32))[None]), refv)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def _labels(seed, shape=(256, 256)):
    return random_blobs(shape, n_cells=40, radius=(4, 10), seed=seed).astype(np.int32)


@pytest.mark.parametrize('seed', range(3))
def test_targets_match_semantic_iterator(ref, seed):
    """The dataset's targets equal ``SemanticIterator._transform_labels`` with
    the notebook's transforms and kwargs, concatenated in head order."""
    lab = _labels(seed)
    ds = SegmentationDataset(np.zeros((1, 1) + lab.shape, np.float32), lab[None, None])
    port = ds._transform_labels(lab[None])
    refs = _quiet(ref.generators.transform_labels, lab[None, ..., None], list(IN_TRANSFORMS), TRANSFORMS_KWARGS)
    assert [r.shape[-1] for r in refs] == [1, 1, 2] == ds.head_channels
    np.testing.assert_array_equal(port, np.concatenate([cf(r)[0] for r in refs], axis=0))


def test_val_targets_are_a_crop_of_the_reference_targets(ref):
    """deepcell crops after computing the targets on the full image; so does the port."""
    lab = random_blobs((512, 512), n_cells=150, radius=(4, 10), seed=11).astype(np.int32)
    ds = SegmentationDataset(np.random.default_rng(0).random((1, 1, 512, 512), np.float32),
                             lab[None, None], dataset_type='val')
    _, y_out = ds[0]
    refs = _quiet(ref.generators.transform_labels, lab[None, ..., None], list(IN_TRANSFORMS), TRANSFORMS_KWARGS)
    full = np.concatenate([cf(r)[0] for r in refs], axis=0)
    np.testing.assert_array_equal(y_out.numpy(), full[:, 128:384, 128:384].astype(np.float32))


# ---------------------------------------------------------------------------
# min_objects
# ---------------------------------------------------------------------------

def test_min_objects_matches_semantic_iterator(ref):
    imgs = [_labels(s, (64, 64)) for s in range(4)]
    empty = np.zeros((64, 64), np.int32)
    one = np.zeros((64, 64), np.int32)
    one[10:20, 10:20] = 7
    full = np.full((64, 64), 3, np.int32)     # no background: deepcell counts 0 objects
    y = np.stack([imgs[0], empty, one, imgs[1], full, imgs[2], empty, imgs[3]])[:, None]
    for min_objects in (MIN_OBJECTS, 2, 30):
        np.testing.assert_array_equal(qc_keep_indices(y, min_objects),
                                      ref.generators.min_objects_keep(cl(y), min_objects))


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@pytest.mark.data
def test_real_images_match_the_notebook_pipeline(ref, dnn_test_slice):
    """Normalisation and targets of real DynamicNuclearNet images, unaugmented."""
    X, y = dnn_test_slice
    ds = SegmentationDataset(X, y, dataset_type='val')
    for i in range(X.shape[0]):
        refx = _quiet(ref.processing.histogram_normalization, cl(X[i:i + 1]).astype(np.uint16))
        np.testing.assert_array_equal(ds._normalize(X[i]), cf(refx)[0])
        refs = _quiet(ref.generators.transform_labels, cl(y[i:i + 1]), list(IN_TRANSFORMS), TRANSFORMS_KWARGS)
        np.testing.assert_array_equal(ds._transform_labels(y[i]), np.concatenate([cf(r)[0] for r in refs], 0))
