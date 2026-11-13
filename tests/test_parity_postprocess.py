"""Parity tests: the inference pipeline (``torch_dnn.postprocess_utils``) vs
deepcell-toolbox and deepcell-tf's NuclearSegmentation application.

The port is channels-first ``(B, C, H, W)``; the reference is channels-last
``(B, H, W, C)``. Every comparison moves the channel axis explicitly.

NuclearSegmentation's steps are: resize to model_mpp, ``histogram_normalization``
(default kernel), tile, run the prediction model (which keeps only the inner-
and outer-distance heads), untile, ``deep_watershed([inner, outer], ...)`` with
the 1.1 config, resize back.
"""

import warnings

import numpy as np
import pytest
import scipy.ndimage as nd

from torch_dnn import postprocess_utils as pp
from torch_dnn.dnn import DEFAULT_POSTPROCESS_KWARGS

from tests.conftest import random_blobs

MODEL_SHAPE = (256, 256)

# deepcell-tf nuclear_segmentation.py, CONFIGS['1.1'], as NuclearSegmentation passes it.
TF_NUCLEAR_SEGMENTATION_KWARGS = dict(radius=10, maxima_threshold=0.1, interior_threshold=0.01,
                                      exclude_border=False, small_objects_threshold=0, min_distance=10)


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


def _app(ref, name):
    """A transcribed deepcell application helper from ``tests/_reference.py``, or skip."""
    app = getattr(ref, 'app', None)
    if app is None or not hasattr(app, name):
        pytest.skip(f'tests/_reference.py does not provide app.{name}')
    return getattr(app, name)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kernel_size', [None, 32])
def test_dnn_preprocess_matches_histogram_normalization(ref, kernel_size):
    """NuclearSegmentation's preprocessing_fn is histogram_normalization with no extra kwargs."""
    x = _image((3, 96, 80, 1), blank=(1, 0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kw = {} if kernel_size is None else {'kernel_size': kernel_size}
        port = pp.dnn_preprocess(cf(x), **kw)
        refv = ref.processing.histogram_normalization(x, **kw)
    np.testing.assert_array_equal(cl(port), refv)
    assert port.dtype == np.float32
    assert (port[1] == 0).all()  # constant image -> zeros


def test_dnn_preprocess_does_not_threshold(ref):
    """Unlike Mesmer, there is no percentile threshold before CLAHE."""
    x = _image((1, 64, 64, 1), seed=3)
    x[0, 10, 10, 0] = 1e4  # one very bright pixel
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        port = pp.dnn_preprocess(cf(x))
        refv = ref.processing.histogram_normalization(x)
    np.testing.assert_array_equal(cl(port), refv)


def test_dnn_preprocess_rejects_wrong_rank():
    with pytest.raises(ValueError, match='4D'):
        pp.dnn_preprocess(np.zeros((64, 64), np.float32))


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

TILE_SHAPES = [
    (1, 600, 520, 1),
    (2, 256, 600, 1),   # one side exactly 256: spline blending is skipped (same in both)
    (1, 300, 300, 1),
    (3, 512, 512, 1),
    (1, 512, 512, 4),   # the network output has 4 channels
]


@pytest.mark.parametrize('pad_mode', ['constant', 'reflect'])
@pytest.mark.parametrize('shape', TILE_SHAPES)
def test_tile_image_matches(ref, shape, pad_mode):
    x = np.random.default_rng(0).random(shape).astype(np.float32)
    tiles_p, info_p = pp.tile_image(cf(x), MODEL_SHAPE, stride_ratio=0.75, pad_mode=pad_mode)
    tiles_r, info_r = ref.utils.tile_image(x, MODEL_SHAPE, stride_ratio=0.75, pad_mode=pad_mode)

    np.testing.assert_array_equal(cl(tiles_p), tiles_r)
    assert set(info_p) == set(info_r)
    for key in info_r:
        if key == 'image_shape':  # the port stores it channels-first
            assert tuple(info_p[key]) == tuple(cf(np.empty(info_r[key])).shape)
        else:
            assert info_p[key] == info_r[key], key


@pytest.mark.parametrize('shape', TILE_SHAPES)
def test_untile_image_matches(ref, shape):
    x = np.random.default_rng(1).random(shape).astype(np.float32)
    tiles_p, info_p = pp.tile_image(cf(x), MODEL_SHAPE, stride_ratio=0.75)
    tiles_r, info_r = ref.utils.tile_image(x, MODEL_SHAPE, stride_ratio=0.75)

    out_p = pp.untile_image(tiles_p, info_p)
    out_r = ref.utils.untile_image(tiles_r, info_r)
    np.testing.assert_array_equal(cl(out_p), out_r)
    # The spline windows sum to 1, so tiling then untiling is (nearly) lossless.
    np.testing.assert_allclose(cl(out_p), x, atol=1e-3)


@pytest.mark.parametrize('shape', [
    (1, 200, 200, 1),   # smaller than the model: reflect-pad
    (1, 256, 200, 1),   # one side exactly 256, other smaller
    (1, 201, 150, 1),   # odd padding
    (2, 300, 300, 1),   # larger: tile
])
def test_tile_input_and_untile_output_match(ref, shape):
    tile_input, untile_output = _app(ref, 'tile_input'), _app(ref, 'untile_output')
    x = np.random.default_rng(2).random(shape).astype(np.float32)
    tiles_p, info_p = pp.tile_input(cf(x), MODEL_SHAPE, pad_mode='reflect')
    tiles_r, info_r = tile_input(x, MODEL_SHAPE, pad_mode='reflect')
    np.testing.assert_array_equal(cl(tiles_p), tiles_r)

    out_p = pp.untile_output(tiles_p, info_p)
    out_r = untile_output(tiles_r, info_r, MODEL_SHAPE)
    np.testing.assert_array_equal(cl(out_p), out_r)
    assert out_p.shape == cf(x).shape


@pytest.mark.parametrize('shape', [(1, 1, 200, 600), (2, 1, 600, 200), (1, 1, 100, 300), (1, 1, 255, 257)])
def test_tile_input_one_side_small_one_side_large(shape):
    """Deliberate improvement over the reference (which pads the long side too and
    crashes): only the short side is padded, the long side is tiled, every tile is
    model-sized, and untiling gives back the input."""
    x = np.random.default_rng(3).random(shape).astype(np.float32)
    tiles, info = pp.tile_input(x, MODEL_SHAPE)
    assert tiles.shape[-2:] == MODEL_SHAPE
    np.testing.assert_allclose(pp.untile_output(tiles, info), x, atol=1e-3)


# ---------------------------------------------------------------------------
# Resizing (NuclearSegmentation's model_mpp is 0.65)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('image_mpp', [0.325, 0.5, 0.55, 1.0])
def test_resize_input_matches(ref, image_mpp):
    resize_input = _app(ref, 'resize_input')
    x = np.random.default_rng(4).random((2, 1, 203, 190)).astype(np.float32)
    port = pp.resize_input(x, image_mpp, 0.65)
    refv = resize_input(cl(x), image_mpp, 0.65)
    np.testing.assert_array_equal(cl(port), refv)


def test_resize_input_same_or_none_mpp_is_noop():
    x = np.random.default_rng(5).random((1, 1, 100, 90)).astype(np.float32)
    assert pp.resize_input(x, 0.65, 0.65) is x
    assert pp.resize_input(x, None, 0.65) is x


@pytest.mark.parametrize('dtype', [np.uint16, np.float32])
def test_resize_input_single_channel_uses_cv2_like_deepcell(ref, dtype):
    """deepcell-toolbox resizes single-channel images with cv2 INTER_LINEAR (no
    anti-aliasing), so a raw uint16 frame is resized exactly as in NuclearSegmentation."""
    x = (np.random.default_rng(6).random((2, 1, 300, 280)) * 4000).astype(dtype)
    for image_mpp in (0.5, 0.8):
        port = pp.resize_input(x, image_mpp, 0.65)
        refv = ref.utils.resize(cl(x), tuple(int(s * image_mpp / 0.65) for s in x.shape[2:]),
                                data_format='channels_last')
        assert port.dtype == x.dtype
        np.testing.assert_array_equal(cl(port), refv)


@pytest.mark.parametrize('out_hw', [(150, 140), (400, 380), (406, 380)])
def test_resize_output_matches(ref, out_hw):
    """Masks are resized back with nearest neighbour (cv2 for one channel, as in deepcell),
    so no new labels are invented. The port also accepts float64 and int64 masks."""
    resize_output = _app(ref, 'resize_output')
    lab = random_blobs((101, 95), 20, seed=0)[None, None]
    refv = resize_output(cl(lab.astype(np.int32)), (1, *out_hw, 1))
    for dtype in (np.int32, np.int64, np.float64):
        port = pp.resize_output(lab.astype(dtype), (1, 1, *out_hw))
        assert port.shape == (1, 1, *out_hw) and port.dtype == dtype
        np.testing.assert_array_equal(cl(port), refv)
    assert set(np.unique(refv)) <= set(np.unique(lab))


def test_resize_multichannel_still_uses_skimage(ref):
    """The 4-channel network output (only resized by callers) follows the toolbox's skimage path."""
    x = np.random.default_rng(7).random((1, 4, 101, 95)).astype(np.float32)
    from torch_dnn.utils import resize
    np.testing.assert_array_equal(cl(resize(x, (150, 140), data_format='channels_first')),
                                  ref.utils.resize(cl(x), (150, 140), data_format='channels_last'))


# ---------------------------------------------------------------------------
# deep_watershed and dnn_postprocess
# ---------------------------------------------------------------------------

def _fake_dnn_output(ref, labels, seed):
    """A plausible (4, H, W) DynamicNuclearNet output for one label image: smoothed inner
    and outer distance targets and fgbg probabilities, plus a little noise."""
    rng = np.random.default_rng(seed)
    tu = ref.transform_utils
    inner = tu.inner_distance_transform_2d(labels, alpha='auto')
    outer = tu.outer_distance_transform_2d(labels)
    inner = nd.gaussian_filter(inner, 1) + 0.02 * rng.random(inner.shape)
    outer = nd.gaussian_filter(outer, 1) + 0.02 * rng.random(outer.shape)
    fg = np.clip(nd.gaussian_filter((labels > 0).astype(float), 1) + 0.02 * rng.random(labels.shape), 0, 1)
    return np.stack([inner, outer, 1 - fg, fg])


@pytest.fixture(scope='module')
def fake_model_output(ref):
    """A DynamicNuclearNet output, (B=2, 4, 128, 128): inner, outer, background, foreground."""
    batch = [_fake_dnn_output(ref, random_blobs((128, 128), 25, radius=(3, 7), seed=10 + b), seed=b)
             for b in range(2)]
    return np.stack(batch).astype(np.float32)


WATERSHED_KWARGS = [
    {},
    dict(pixel_expansion=1),
    dict(pixel_expansion=2, maxima_smooth=1),
    dict(maxima_algorithm='peak_local_max'),
    dict(maxima_algorithm='peak_local_max', exclude_border=True),
    dict(radius=2, interior_threshold=0.1, small_objects_threshold=15, fill_holes_threshold=15,
         label_erosion=1),
]


@pytest.mark.parametrize('kwargs', WATERSHED_KWARGS, ids=lambda k: ','.join(f'{a}={b}' for a, b in k.items()))
def test_deep_watershed_matches(ref, fake_model_output, kwargs):
    two = fake_model_output[:, [0, 1]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        port = pp.deep_watershed(two, maxima_index=0, interior_index=1, **kwargs)
        refv = ref.deep_watershed.deep_watershed([cl(two[:, :1]), cl(two[:, 1:])],
                                                 maxima_index=0, interior_index=1, **kwargs)
    assert port.max() > 5  # the fixture really produces objects
    np.testing.assert_array_equal(cl(port), refv)


def test_dnn_postprocess_matches_nuclear_segmentation(ref, fake_model_output):
    """``dnn_postprocess`` at the app's defaults gives NuclearSegmentation's masks:
    ``deep_watershed([inner, outer], **CONFIGS['1.1'])`` on the prediction model's two
    outputs (min_distance=10 is a deprecated alias of radius=10)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        port = pp.dnn_postprocess(fake_model_output, **DEFAULT_POSTPROCESS_KWARGS)
        refv = ref.deep_watershed.deep_watershed(
            [cl(fake_model_output[:, 0:1]), cl(fake_model_output[:, 1:2])], **TF_NUCLEAR_SEGMENTATION_KWARGS)
    assert port.shape == (2, 1, 128, 128)
    assert port.max() > 5
    np.testing.assert_array_equal(cl(port), refv)


def test_dnn_postprocess_uses_inner_then_outer_distance(fake_model_output):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        default = pp.dnn_postprocess(fake_model_output)
        explicit = pp.deep_watershed(fake_model_output, maxima_index=0, interior_index=1)
        on_fg = pp.dnn_postprocess(fake_model_output, interior_index=3)
    np.testing.assert_array_equal(default, explicit)
    assert pp.DNN_HEAD_INDICES == {'maxima_index': 0, 'interior_index': 1}
    assert not np.array_equal(default, on_fg)


def test_deep_watershed_defaults_match(ref, fake_model_output):
    two = fake_model_output[:, [0, 1]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        port = pp.deep_watershed(two)
        refv = ref.deep_watershed.deep_watershed([cl(two[:, :1]), cl(two[:, 1:])])
    np.testing.assert_array_equal(cl(port), refv)


def test_deep_watershed_rejects_invalid_algorithm(fake_model_output):
    with pytest.raises(ValueError):
        pp.deep_watershed(fake_model_output[:, [0, 1]], maxima_algorithm='not-an-algorithm')


@pytest.mark.parametrize('old, new, value', [
    ('min_distance', 'radius', 3),
    ('distance_threshold', 'interior_threshold', 0.3),
    ('detection_threshold', 'maxima_threshold', 0.2),
])
def test_deep_watershed_deprecated_alias_value_is_used(ref, fake_model_output, old, new, value):
    """As in the reference, the deprecated alias's value wins over the new name, with a warning."""
    two = fake_model_output[:, [0, 1]]
    with pytest.warns(DeprecationWarning, match=old):
        via_alias = pp.deep_watershed(two, **{old: value})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        via_new = pp.deep_watershed(two, **{new: value})
        refv = ref.deep_watershed.deep_watershed([cl(two[:, :1]), cl(two[:, 1:])], **{old: value})
    np.testing.assert_array_equal(via_alias, via_new)
    np.testing.assert_array_equal(cl(via_alias), refv)


def test_deep_watershed_needs_indices_for_more_than_two_channels(fake_model_output):
    """deepcell's default interior_index=-1 would silently read the foreground channel."""
    with pytest.raises(ValueError, match='maxima_index'):
        pp.deep_watershed(fake_model_output)


@pytest.mark.parametrize('kwargs', [dict(maxima_index=4, interior_index=1), dict(maxima_index=0, interior_index=-5)])
def test_deep_watershed_rejects_out_of_range_index(fake_model_output, kwargs):
    with pytest.raises(ValueError, match='out of range'):
        pp.deep_watershed(fake_model_output, **kwargs)


def test_deep_watershed_rejects_wrong_rank(fake_model_output):
    with pytest.raises(ValueError, match='channels-first'):
        pp.deep_watershed(fake_model_output[0], maxima_index=0, interior_index=1)
