"""Tests for the scoring in ``torch_dnn.eval``.

The eval script must report pooled (micro-averaged) recall, precision and F1,
as deepcell's ``Metrics.summarize_object_metrics_df`` does.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from torch_dnn import eval as ev

from tests.conftest import HAND_BUILT_CASES


def _evaluate(yp, yt):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ev.evaluate(yp, yt)


def test_evaluate_is_pooled(random_label_pairs):
    yt = np.stack([p[0] for p in random_label_pairs[:8]])
    yp = np.stack([p[1] for p in random_label_pairs[:8]])
    per_image, s = _evaluate(yp, yt)
    assert len(per_image) == 8 and s['n_images'] == 8
    correct, n_true, n_pred = (per_image[k].sum() for k in ('correct_detections', 'n_true', 'n_pred'))
    r, p = correct / n_true, correct / n_pred
    assert s['recall'] == round(r, 4)
    assert s['precision'] == round(p, 4)
    assert s['f1'] == round(2 * r * p / (r + p), 4)
    assert s['n_true'] == n_true and s['correct_detections'] == correct
    # The per-image means are reported separately.
    assert s['mean_per_image_recall'] == pytest.approx(per_image['recall'].mean())
    assert s['mean_per_image_f1'] == pytest.approx(per_image['f1'].mean())


def test_pooled_differs_from_mean_of_images():
    """One image with 1 cell (perfect), one with 9 cells (all missed): pooled
    recall is 1/10, while the mean of per-image recall is 0.5."""
    yt_a, _ = HAND_BUILT_CASES['perfect']
    big = np.zeros_like(yt_a)
    for k in range(9):
        r, c = divmod(k, 3)
        big[2 + 6 * r:6 + 6 * r, 2 + 6 * c:6 + 6 * c] = k + 1
    yt = np.stack([yt_a, big])
    yp = np.stack([yt_a, np.zeros_like(big)])
    _, s = _evaluate(yp, yt)
    assert s['recall'] == 0.1
    assert s['mean_per_image_recall'] == 0.5


def test_summarize_by_group(random_label_pairs):
    yt = np.stack([p[0] for p in random_label_pairs[:6]])
    yp = np.stack([p[1] for p in random_label_pairs[:6]])
    per_image, overall = _evaluate(yp, yt)
    per_image.insert(0, 'specimen', ['HeLa', 'HeLa', 'RAW', 'RAW', 'RAW', 'HEK'])
    out = ev.summarize_by_group(per_image, 'specimen')
    assert list(out.index) == ['all', 'specimen=HEK', 'specimen=HeLa', 'specimen=RAW']
    assert out.loc['all', 'f1'] == overall['f1']
    assert out.loc['specimen=RAW', 'n_images'] == 3
    _, raw = _evaluate(yp[2:5], yt[2:5])
    assert out.loc['specimen=RAW', 'f1'] == raw['f1']


def test_image_mpp_or_none():
    assert ev.image_mpp_or_none(0.65) == 0.65
    assert ev.image_mpp_or_none(np.float64(0.55)) == 0.55
    assert ev.image_mpp_or_none(np.nan) is None


class _FakeApp:
    """Stands in for ``DNN``: returns the ground truth eroded by one pixel, and records
    the ``image_mpp`` it was given."""

    def __init__(self, y):
        self.y = y
        self.mpps = []
        self.calls = 0

    def predict(self, x, image_mpp, batch_size=4, verbose=True):
        from torch_dnn.utils import erode_edges
        assert x.shape == (1, 1, 64, 64) and verbose is False
        self.mpps.append(image_mpp)
        out = erode_edges(self.y[self.calls, 0].astype(int), 1)[None, None]
        self.calls += 1
        return out


@pytest.fixture
def tiny_split(tmp_path, random_label_pairs):
    """A 3-image Zarr split in the layout torch_dnn.preprocess writes."""
    import zarr
    y = np.stack([p[0][:64, :64] for p in random_label_pairs[:3]])[:, None].astype(np.float32)
    X = np.random.default_rng(0).random(y.shape).astype(np.float32)
    meta = np.array([('a', 'e1', 0.65, 'HeLa'), ('b', 'e1', np.nan, 'HeLa'), ('c', 'e2', 0.55, 'RAW')],
                    dtype=[('filename', '<U8'), ('experiment', '<U8'), ('pixel_size', '<f8'), ('specimen', '<U8')])
    path = tmp_path / 'split.zarr'
    z = zarr.open(str(path), mode='w')
    z['X'] = X
    z['y'] = y
    z['meta'] = meta
    return path, y


def test_run_evaluation_reads_images_and_mpps(tiny_split):
    path, y = tiny_split
    app = _FakeApp(y)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        per_image = ev.run_evaluation(app, path)
    assert app.mpps == [0.65, None, 0.55]  # a missing pixel size means no rescaling
    assert list(per_image['image']) == [0, 1, 2]
    assert list(per_image['specimen']) == ['HeLa', 'HeLa', 'RAW']
    assert (per_image['n_true'] > 0).all()
    assert per_image['recall'].between(0, 1).all()

    app = _FakeApp(y)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        per_image = ev.run_evaluation(app, path, use_mpp=False, limit=2)
    assert app.mpps == [None, None] and len(per_image) == 2
    assert isinstance(ev.summarize_by_group(per_image), pd.DataFrame)
