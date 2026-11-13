"""Tests for the training data pipeline: ``torch_dnn.augmentation``,
``torch_dnn.loaders`` and ``torch_dnn.preprocess``.

Copied from torch-mesmer's ``tests/test_data_pipeline.py`` and adapted to the
DynamicNuclearNet recipe (one channel in, targets inner distance / outer
distance / fgbg, CLAHE with default kernel, zoom 0.75, ``min_objects=1``).
"""

import random
import zipfile

import numpy as np
import pytest
import scipy.ndimage as nd
import skimage.morphology
import torch
import torch.nn.functional as F

from torch_dnn import loaders, preprocess
from torch_dnn.augmentation import MultiTransform
from torch_dnn.loaders import SegmentationDataset, create_data_loaders
from torch_dnn.transforms import transform_masks
from torch_dnn.utils import histogram_normalization, percentile_threshold

from tests.conftest import DNN_DATA_DIR, random_blobs


def _fixed_transform(shape, crop_h, crop_w, top=0, left=0, angle=0.0, hflip=False, vflip=False, crop_size=256):
    t = MultiTransform(shape, dataset_type='train', crop_size=crop_size)
    t.crop_h, t.crop_w, t.top, t.left = crop_h, crop_w, top, left
    t.angle, t.do_hflip, t.do_vflip = angle, hflip, vflip
    return t


# ---------------------------------------------------------------------------
# MultiTransform
# ---------------------------------------------------------------------------

def test_val_is_identity_at_crop_size():
    x = torch.rand(3, 256, 256)
    t = MultiTransform((256, 256), dataset_type='val')
    torch.testing.assert_close(t(x, ['bilinear', 'nearest', 'nearest']), x, rtol=0, atol=0)


def test_val_centre_crops_larger_images():
    x = torch.rand(1, 512, 512)
    out = MultiTransform((512, 512), dataset_type='val')(x, ['bilinear'])
    torch.testing.assert_close(out, x[:, 128:384, 128:384], rtol=0, atol=0)


def test_zoom_only_is_a_plain_resize_without_antialiasing():
    # Angle 0, no flips: the single resampling must equal crop + bilinear
    # resize with align_corners=False and no antialiasing.
    x = torch.rand(2, 512, 512)
    t = _fixed_transform((512, 512), crop_h=330, crop_w=190, top=40, left=100)
    out = t(x, ['bilinear', 'bilinear'])
    ref = F.interpolate(x[None, :, 40:370, 100:290], size=(256, 256), mode='bilinear',
                        align_corners=False, antialias=False)[0]
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-4)


def test_zoom_only_nearest_matches_nearest_exact():
    x = torch.randint(0, 50, (1, 512, 512)).float()
    t = _fixed_transform((512, 512), crop_h=300, crop_w=200, top=10, left=20)
    out = t(x, ['nearest'])
    ref = F.interpolate(x[None, :, 10:310, 20:220], size=(256, 256), mode='nearest-exact')[0]
    assert (out != ref).float().mean() < 0.01  # differences only at exact rounding ties


@pytest.mark.parametrize('mode', ['bilinear', 'nearest'])
def test_rotation_is_counter_clockwise(mode):
    x = torch.rand(1, 256, 256)
    t = _fixed_transform((256, 256), crop_h=256, crop_w=256, angle=90.0)
    torch.testing.assert_close(t(x, [mode]), torch.rot90(x, 1, dims=(1, 2)), rtol=0, atol=1e-4)


def test_flips():
    x = torch.rand(1, 256, 256)
    out = _fixed_transform((256, 256), 256, 256, hflip=True, vflip=True)(x, ['nearest'])
    torch.testing.assert_close(out, x.flip(-1).flip(-2), rtol=0, atol=0)


def test_rotation_corners_use_nearest_fill():
    x = torch.ones(2, 256, 256)
    out = _fixed_transform((256, 256), 256, 256, angle=45.0)(x, ['bilinear', 'nearest'])
    torch.testing.assert_close(out, x, rtol=0, atol=1e-6)


def test_crop_larger_than_image_is_edge_padded():
    x = torch.rand(1, 256, 256)
    t = _fixed_transform((256, 256), crop_h=300, crop_w=300, top=-20, left=-30)
    out = t(x, ['nearest'])
    assert out.shape == (1, 256, 256)
    assert torch.isfinite(out).all() and (out > 0).all()


def test_zoom_default_is_the_dnn_range_and_anisotropic():
    random.seed(0)
    hw = np.array([[t.crop_h, t.crop_w] for t in
                   (MultiTransform((512, 512), 'train') for _ in range(2000))])
    lo, hi = 0.75 * 256, 256 / 0.75
    assert hw.min() >= np.floor(lo) and hw.max() <= np.ceil(hi)
    assert hw.min() < lo + 3 and hw.max() > hi - 3
    assert (hw[:, 0] != hw[:, 1]).mean() > 0.9
    assert abs(np.corrcoef(hw.T)[0, 1]) < 0.1


def test_crop_position_uses_the_image_shape():
    random.seed(0)
    for _ in range(200):
        t = MultiTransform((400, 1000), 'train')
        assert 0 <= t.top <= 400 - t.crop_h and 0 <= t.left <= 1000 - t.crop_w


def test_rejects_bad_modes_and_shapes():
    t = MultiTransform((64, 64), 'train', crop_size=32)
    with pytest.raises(ValueError, match='unknown interpolation'):
        t(torch.rand(1, 64, 64), ['bicubic'])
    with pytest.raises(ValueError, match='interpolation modes'):
        t(torch.rand(2, 64, 64), ['bilinear'])
    with pytest.raises(ValueError, match='drawn for'):
        t(torch.rand(1, 32, 64), ['bilinear'])


# ---------------------------------------------------------------------------
# SegmentationDataset
# ---------------------------------------------------------------------------

def _sample(seed=0, shape=(512, 512), n_cells=120):
    """One nuclear label image, plus a 1-channel image that is bright exactly
    on the foreground, so the image and the fgbg foreground channel should
    overlap after any augmentation."""
    labels = random_blobs(shape, n_cells=n_cells, radius=(6, 12), seed=seed)
    labels = np.where(nd.binary_erosion(labels > 0, skimage.morphology.disk(1)), labels, 0)
    rng = np.random.default_rng(seed)
    X = (labels > 0).astype(np.float32)[None] * 100 + rng.random((1,) + shape, np.float32)
    return X[None], labels[None, None].astype(np.int32)


def _iou(a, b):
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


@pytest.mark.parametrize('seed', range(4))
def test_augmented_sample_is_one_hot_and_aligned(seed):
    X, y = _sample()
    ds = SegmentationDataset(X, y, dataset_type='train', semantic_heads=[1, 1, 2])
    random.seed(seed)
    x_out, y_out = ds[0]
    x_out, y_out = x_out.numpy(), y_out.numpy()

    assert x_out.shape == (1, 256, 256) and y_out.shape == (4, 256, 256)
    inner, outer, bg, fg = y_out
    assert set(np.unique(y_out[2:])) <= {0.0, 1.0}
    np.testing.assert_array_equal(bg + fg, 1)  # one-hot [background, foreground]

    # Image and labels moved together: the bright image regions are the foreground.
    assert _iou(x_out[0] > 0.5, fg > 0) > 0.95
    # Both distance maps live inside the cells and are in [0, 1].
    for d in (inner, outer):
        assert d.min() >= 0 and d.max() <= 1 + 1e-6
        assert (d[bg == 1] < 0.05).mean() > 0.97


def test_augmentation_is_seeded():
    X, y = _sample()
    ds = SegmentationDataset(X, y, dataset_type='train')
    random.seed(3)
    a = ds[0]
    random.seed(3)
    b = ds[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_val_sample_is_the_centre_crop_of_the_full_image_targets():
    """Normalisation and targets are computed on the full 512 px image, then
    the centre 256 px are cut out (no augmentation)."""
    X, y = _sample()
    ds = SegmentationDataset(X, y, dataset_type='val')
    x_out, y_out = ds[0]
    c = slice(128, 384)
    np.testing.assert_allclose(x_out.numpy(), ds._normalize(X[0])[:, c, c], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(y_out.numpy(), ds._transform_labels(y[0])[:, c, c])


def test_normalisation_is_default_clahe_without_percentile_clip():
    rng = np.random.default_rng(0)
    X = nd.gaussian_filter(rng.gamma(2.0, 10.0, size=(1, 1, 512, 512)), sigma=(0, 0, 2, 2)).astype(np.float32)
    X[0, 0, 10, 10] = 3 * X.max()   # one hot pixel: a percentile clip would change the result
    ds = SegmentationDataset(X, np.zeros_like(X, dtype=np.int32))
    expected = histogram_normalization(X)[0]
    np.testing.assert_array_equal(ds._normalize(X[0]), expected)
    assert not np.array_equal(expected, histogram_normalization(percentile_threshold(X, 99.9))[0])
    assert not np.array_equal(expected, histogram_normalization(X, kernel_size=128)[0])
    assert expected.dtype == np.float32 and 0 <= expected.min() and expected.max() <= 1


def test_transform_kwargs_are_used():
    # The kwargs reach transform_masks instead of a hard-coded dict.
    X, y = _sample(shape=(128, 128), n_cells=20)
    kw = {'outer-distance': {'erosion_width': 2}, 'inner-distance': {'erosion_width': 1, 'alpha': 0.5}}
    ds = SegmentationDataset(X, y, dataset_type='val', crop_size=128, transforms_kwargs=kw)
    targets = ds._transform_labels(y[0])
    lab = y[0]
    np.testing.assert_array_equal(targets[0], transform_masks(lab, 'inner-distance', unbatched=True,
                                                              erosion_width=1, alpha=0.5)[0, 0])
    np.testing.assert_array_equal(targets[1], transform_masks(lab, 'outer-distance', unbatched=True,
                                                              erosion_width=2)[0, 0])


def test_default_kwargs_are_the_recipe():
    assert loaders.IN_TRANSFORMS == ('inner-distance', 'outer-distance', 'fgbg')
    assert loaders.TRANSFORMS_KWARGS == {'outer-distance': {'erosion_width': 1},
                                         'inner-distance': {'alpha': 'auto', 'beta': 1, 'erosion_width': 0}}
    assert loaders.ZOOM_MIN == 0.75 and loaders.CLAHE_KERNEL_SIZE is None
    assert loaders.MIN_OBJECTS == 1 and loaders.MIN_OBJECT_SIZE == 0


def test_default_layout_matches_the_model_output():
    X, y = _sample(shape=(64, 64), n_cells=10)
    ds = SegmentationDataset(X, y, crop_size=64)
    assert ds.head_channels == [1, 1, 2]
    assert ds.interpolation == ['bilinear', 'bilinear', 'bilinear', 'nearest', 'nearest']


def test_channel_layout_follows_in_transforms():
    X, y = _sample(shape=(256, 256), n_cells=40)
    ds = SegmentationDataset(X, y, in_transforms=['fgbg', 'pixelwise'], crop_size=128)
    assert ds.head_channels == [2, 3]
    assert ds.interpolation == ['bilinear'] + ['nearest'] * 5
    random.seed(0)
    _, y_out = ds[0]
    assert y_out.shape == (5, 128, 128)
    np.testing.assert_array_equal(y_out[0:2].sum(0).numpy(), 1)
    np.testing.assert_array_equal(y_out[2:5].sum(0).numpy(), 1)


def test_semantic_heads_mismatch_is_an_error():
    X, y = _sample(shape=(64, 64), n_cells=10)
    with pytest.raises(ValueError, match='semantic_heads'):
        SegmentationDataset(X, y, semantic_heads=[1, 3, 1, 3])


class _RecordingArray:
    """Array stand-in that records how it is indexed."""

    def __init__(self, a):
        self.a, self.shape, self.ndim, self.keys = a, a.shape, a.ndim, []

    def __getitem__(self, key):
        self.keys.append(key)
        return self.a[key]


def test_dataset_reads_one_image_at_a_time():
    # No fancy-indexing, which would load the whole zarr array.
    X, y = _sample(shape=(64, 64), n_cells=10)
    X, y = np.repeat(X, 30, axis=0), np.repeat(y, 30, axis=0)
    rx, ry = _RecordingArray(X), _RecordingArray(y)
    ds = SegmentationDataset(rx, ry, dataset_type='val', crop_size=64, min_objects=1)
    ds[3]
    assert len(ds) == 30
    assert all(isinstance(k, (int, tuple)) for k in rx.keys + ry.keys)
    assert all(isinstance(k if isinstance(k, int) else k[0], int) for k in rx.keys + ry.keys)


# ---------------------------------------------------------------------------
# min_objects filter
# ---------------------------------------------------------------------------

def test_count_objects_is_deepcells_unique_count():
    lab = random_blobs((96, 96), n_cells=12, seed=4)
    assert loaders.count_objects(lab) == len(np.unique(lab)) - 1
    assert loaders.count_objects(np.zeros((8, 8), np.int32)) == 0


def test_count_objects_with_min_size_matches_skimage():
    import warnings
    lab = random_blobs((128, 128), n_cells=60, radius=(1, 6), seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        kept = skimage.morphology.remove_small_objects(lab, min_size=15)
    np.testing.assert_array_equal(loaders.remove_small_objects(lab, 15), kept)
    assert loaders.count_objects(lab, 15) == len(np.unique(kept)) - 1


def test_min_objects_drops_empty_images_from_train_and_val():
    good = random_blobs((64, 64), n_cells=10, seed=0)[None]
    split = {'X': np.random.default_rng(0).random((4, 1, 64, 64), np.float32),
             'y': np.stack([good, good, np.zeros_like(good), good]).astype(np.int32)}
    train, val = create_data_loaders(split, split, crop_size=64, batch_size=1, num_workers=0)
    assert len(train.dataset) == 3
    assert len(val.dataset) == 3
    np.testing.assert_array_equal(val.dataset.indices, [0, 1, 3])


def test_max_images_limits_each_split():
    good = random_blobs((64, 64), n_cells=10, seed=0)[None]
    split = {'X': np.zeros((10, 1, 64, 64), np.float32), 'y': np.repeat(good[None], 10, 0).astype(np.int32)}
    train, val = create_data_loaders(split, split, crop_size=64, batch_size=2, num_workers=0,
                                     max_train_images=6, max_val_images=4)
    assert len(train.dataset) == 6 and len(val.dataset) == 4


# ---------------------------------------------------------------------------
# preprocess: npz -> zarr
# ---------------------------------------------------------------------------

def _fake_npz(path, n=5, h=32, w=48):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 65535, size=(n, h, w, 1), dtype=np.uint16)
    y = rng.integers(0, 9, size=(n, h, w, 1)).astype(np.int32)
    meta = np.empty((n + 1, 6), dtype=object)
    meta[0] = preprocess.META_COLUMNS
    for i in range(n):
        meta[i + 1] = [f'movie{i}.trk', 'exp', [0.65, 0.55, None, np.nan, 0.65][i % 5],
                       'Not screened', [None, '6 min'][i % 2], 'HeLa']
    np.savez_compressed(path, X=X, y=y, meta=meta)
    return X, y, meta


@pytest.mark.parametrize('block', [1, 2, 32])
def test_preprocess_round_trip(tmp_path, block):
    X, y, meta = _fake_npz(tmp_path / 'val.npz')
    out = preprocess.convert_to_zarr(tmp_path / 'val.npz', tmp_path / 'out', block=block)
    import zarr
    z = zarr.open(str(out), mode='r')
    assert z['X'].dtype == np.float32 and z['y'].dtype == np.int32
    assert z['X'].chunks == (1, 1, 32, 48)
    np.testing.assert_array_equal(z['X'][:], np.moveaxis(X, -1, 1).astype(np.float32))
    np.testing.assert_array_equal(z['y'][:], np.moveaxis(y, -1, 1))
    m = z['meta'][:]
    assert list(m.dtype.names) == list(preprocess.META_COLUMNS)
    assert list(m['filename']) == [f'movie{i}.trk' for i in range(5)]
    np.testing.assert_array_equal(m['pixel_size'], [0.65, 0.55, np.nan, np.nan, 0.65])
    assert list(m['time_step']) == ['', '6 min', '', '6 min', '']
    assert z.attrs['split'] == 'val'
    assert not (tmp_path / 'out' / 'val.zarr.partial').exists()


def test_preprocess_does_not_overwrite_by_default(tmp_path):
    _fake_npz(tmp_path / 'test.npz')
    out = preprocess.convert_to_zarr(tmp_path / 'test.npz', tmp_path)
    marker = out / 'keep_me'
    marker.write_text('x')
    preprocess.convert_to_zarr(tmp_path / 'test.npz', tmp_path)
    assert marker.exists()
    preprocess.convert_to_zarr(tmp_path / 'test.npz', tmp_path, overwrite=True)
    assert not marker.exists()


def test_preprocess_rejects_unexpected_meta_header():
    meta = np.array([['a', 'b'], ['1', '2']], dtype=object)
    with pytest.raises(ValueError, match='header'):
        preprocess.meta_to_structured(meta)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def _open_real_split(split):
    import zarr
    path = DNN_DATA_DIR / f'{split}.zarr'
    if not path.exists():
        pytest.skip(f'{path} is missing (run python -m torch_dnn.preprocess)')
    return zarr.open(str(path), mode='r')


@pytest.mark.data
@pytest.mark.parametrize('split, n', [('train', 4950), ('val', 1417), ('test', 717)])
def test_real_zarr_layout(split, n):
    z = _open_real_split(split)
    assert z['X'].shape == (n, 1, 512, 512) and z['X'].dtype == np.float32
    assert z['y'].shape == (n, 1, 512, 512) and z['y'].dtype == np.int32
    assert z['meta'].shape == (n,)
    assert set(z['meta'][:]['specimen']) == {'HeLa', 'PC-3', '3T3', 'HEK293', 'RPE', 'RAW264'}


@pytest.mark.data
def test_real_zarr_first_images_match_npz():
    z = _open_real_split('test')
    src = preprocess.DEFAULT_SOURCE_DIR / 'test.npz'
    if not src.exists():
        pytest.skip(f'{src} is missing')
    with zipfile.ZipFile(src) as zf:
        for name in ('X', 'y'):
            _, rows = next(preprocess._iter_blocks(zf, f'{name}.npy', 4))
            np.testing.assert_array_equal(z[name][:4], np.moveaxis(rows, -1, 1))


@pytest.mark.data
def test_real_batches_have_the_right_shapes_and_ranges():
    z_train, z_val = _open_real_split('train'), _open_real_split('val')
    train, val = create_data_loaders(z_train, z_val, batch_size=4, num_workers=0,
                                     max_train_images=8, max_val_images=8, semantic_heads=[1, 1, 2])
    for loader in (train, val):
        x, y = next(iter(loader))
        assert x.shape == (4, 1, 256, 256) and y.shape == (4, 4, 256, 256)
        assert x.dtype == y.dtype == torch.float32
        assert 0 <= x.min() and x.max() <= 1 and x.std() > 0.05
        assert 0 <= y[:, :2].min() and y[:, :2].max() <= 1 + 1e-6
        assert set(torch.unique(y[:, 2:]).tolist()) <= {0.0, 1.0}
        torch.testing.assert_close(y[:, 2:].sum(1), torch.ones(4, 256, 256))
        assert y[:, 3].mean() > 0.01   # there are nuclei


@pytest.mark.data
@pytest.mark.slow
def test_no_dnn_train_or_val_image_is_empty():
    """deepcell's ``min_objects=1`` keeps every DynamicNuclearNet image."""
    for split in ('train', 'val'):
        y = _open_real_split(split)['y']
        assert len(loaders.qc_keep_indices(y)) == y.shape[0]
