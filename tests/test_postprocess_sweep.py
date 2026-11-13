"""Tests for ``torch_dnn.postprocess_sweep``.

The sweep reuses work between settings, so it must still give exactly the
masks ``DNN.predict`` would: ``deep_watershed`` then ``resize_output``.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as nd

from torch_dnn import postprocess_sweep as ps
from torch_dnn.dnn import DEFAULT_POSTPROCESS_KWARGS
from torch_dnn.metrics import Metrics
from torch_dnn.postprocess_utils import deep_watershed, resize_output


def _synthetic_maps(shape, n_cells=25, seed=0):
    """Smooth blob maps resembling an inner-distance map and an outer-distance map."""
    rng = np.random.default_rng(seed)
    seeds = np.zeros(shape)
    pts = rng.integers(8, np.array(shape) - 8, size=(n_cells, 2))
    seeds[pts[:, 0], pts[:, 1]] = rng.uniform(0.5, 1.0, n_cells)
    maxima = nd.gaussian_filter(seeds, 3)
    maxima = maxima / maxima.max() + rng.normal(0, 0.02, shape)
    interior = nd.gaussian_filter((nd.gaussian_filter(seeds, 6) > 0.002).astype(float), 1.5)
    interior = np.clip(interior + rng.normal(0, 0.05, shape), 0, 1)
    return maxima.astype(np.float32), interior.astype(np.float32)


SETTINGS = [
    {**ps.BASE_KWARGS},
    {**ps.BASE_KWARGS, 'maxima_threshold': 0.05, 'interior_threshold': 0.1, 'maxima_smooth': 1, 'radius': 2},
    {**ps.BASE_KWARGS, 'interior_smooth': 0, 'radius': 3, 'small_objects_threshold': 0},
    {**ps.BASE_KWARGS, 'pixel_expansion': 1, 'fill_holes_threshold': 0, 'small_objects_threshold': 40},
    {**ps.BASE_KWARGS, 'maxima_threshold': 0.05, 'interior_threshold': 0.1, 'maxima_smooth': 1, 'radius': 2,
     'small_objects_threshold': 30, 'fill_holes_threshold': 15},  # shares its watershed with setting 1
]


def test_base_kwargs_are_the_app_defaults():
    """BASE_KWARGS is DNN.predict's defaults with every other deep_watershed default written out."""
    for k, v in DEFAULT_POSTPROCESS_KWARGS.items():
        if k in ps.BASE_KWARGS:
            assert ps.BASE_KWARGS[k] == v, k
    assert set(DEFAULT_POSTPROCESS_KWARGS) - set(ps.BASE_KWARGS) == {'exclude_border'}  # h_maxima ignores it
    assert ps.MODEL_CHANNELS == [0, 1]  # inner distance, outer distance


@pytest.mark.parametrize('model_hw, original_hw', [
    ((256, 256), (256, 256)),   # 0.65 µm/px: no rescale
    ((512, 512), (256, 256)),   # 1.3 µm/px: model output is shrunk back
    ((197, 197), (256, 256)),   # 0.5 µm/px: model output is enlarged back
])
def test_segment_settings_matches_deep_watershed(model_hw, original_hw):
    maxima, interior = _synthetic_maps(model_hw, seed=model_hw[0])
    transforms = np.stack([maxima, interior])[None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = ps.segment_settings(maxima, interior, SETTINGS, original_hw)
        for s, mask in zip(SETTINGS, got):
            ref = deep_watershed(transforms, maxima_index=0, interior_index=1, **s)
            ref = resize_output(ref, (1, 1) + original_hw).astype(int)[0, 0]
            assert mask.shape == original_hw
            assert ref.max() > 3  # the synthetic maps give real objects
            np.testing.assert_array_equal(mask, ref)


def test_pool_counts_matches_metrics_summary(random_label_pairs):
    pairs = random_label_pairs[:6]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        counts = np.array([ps.score_mask(yt, yp) for yt, yp in pairs])
        m = Metrics('t')
        df = m.calc_object_stats(np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs]),
                                 progbar=False)
    ref = m.summarize_object_metrics_df(df)
    got = ps.pool_counts(counts)
    for k in ('recall', 'precision', 'f1'):
        assert round(got[k], 4) == ref[k]
    for k in ps.COUNT_FIELDS:
        assert got[k] == ref[k]
    assert got['mean_per_image_f1'] == pytest.approx(df['f1'].mean())


def test_expand_grid_fills_base_and_merges_duplicates():
    entries = [
        {'label': 'grid', 'grid': {'interior_threshold': [0.03, 0.05]}},
        {'label': 'ref', 'grid': {'interior_threshold': [0.03]}},
        {'label': 'radius', 'grid': {'radius': [3, 10]}},
    ]
    out = ps.expand_grid(entries)
    assert len(out) == 3  # interior 0.03 and radius 3 are both the base setting
    assert out[0][0] == ['grid', 'ref', 'radius']
    assert out[0][1] == ps.BASE_KWARGS
    assert all(set(kw) == set(ps.SWEEP_KEYS) for _, kw in out)
    with pytest.raises(ValueError):
        ps.expand_grid([{'label': 'x', 'grid': {'bogus': [1]}}])


def test_run_sweep_on_a_tiny_cache(tmp_path):
    """The sweep end to end (cache folder -> CSV) on two synthetic images, with one worker."""
    import zarr
    from skimage.measure import label

    cache = tmp_path / 'cache'
    cache.mkdir()
    maps, ys = [], []
    for seed in range(2):
        maxima, interior = _synthetic_maps((128, 128), seed=seed)
        maps.append(np.stack([maxima, interior]))
        ys.append(label(interior > 0.5)[None].astype(np.float32))
    np.save(cache / 'maps_128x128.npy', np.stack(maps))
    pd.DataFrame({'image': [0, 1], 'group': ['128x128'] * 2, 'row': [0, 1],
                  'orig_h': [128, 128], 'orig_w': [128, 128]}).to_csv(cache / 'index.csv', index=False)
    data = tmp_path / 'val.zarr'
    zarr.open(str(data), mode='w')['y'] = np.stack(ys)
    (cache / 'cache_info.json').write_text(json.dumps({'data_path': str(data)}))

    grid = [{'label': 'thr', 'grid': {'interior_threshold': [0.01, 0.5], 'radius': [2]}}]
    df = ps.run_sweep(cache, grid, tmp_path / 'out.csv', workers=1)
    assert len(df) == 2
    assert set(df.columns) >= {'label', 'recall', 'precision', 'f1', *ps.SWEEP_KEYS, *ps.COUNT_FIELDS}
    assert (tmp_path / 'out.csv').exists() and (tmp_path / 'out.per_image.npz').exists()
    # The mask threshold of the ground truth gives the better F1.
    best = df.sort_values('f1').iloc[-1]
    assert best['interior_threshold'] == 0.5
