"""End-to-end tests of ``torch_dnn.dnn.DNN.predict``.

Most tests build the app around a small stand-in network with the real model's
input/output contract, so everything else in ``predict`` (resize, preprocess,
tile, untile, watershed, resize back) runs for real. The ``slow`` tests build
the actual PanopticNet from a freshly saved random checkpoint.
"""

import inspect
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from torch_dnn import dnn as dnn_module
from torch_dnn.dnn import DEFAULT_POSTPROCESS_KWARGS, DNN, MODEL_MPP, default_model_path
from torch_dnn.model import DNN_MODEL_KWARGS, PanopticNet
from torch_dnn.postprocess_utils import deep_watershed

EXAMPLE_IMAGE = Path(__file__).resolve().parents[1] / 'example' / 'example_image.tiff'


class StubDnnNet(nn.Module):
    """(B, 1, H, W) -> (B, 4, H, W) in DynamicNuclearNet's layout: ReLU on the inner and
    outer distance (channels 0 and 1) and softmax on fgbg (2-3)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv(x)
        return torch.cat([torch.relu(y[:, 0:2]), torch.softmax(y[:, 2:4], dim=1)], dim=1)


def make_app(model):
    """A ``DNN`` with the attributes ``__init__`` sets, minus weight loading."""
    app = DNN.__new__(DNN)
    app.device = 'cpu'
    app.model = model.eval()
    app.model_image_shape = (256, 256)
    app.required_rank = 4
    app.required_channels = 1
    app.model_mpp = MODEL_MPP
    return app


@pytest.fixture(scope='module')
def app():
    return make_app(StubDnnNet())


@pytest.fixture
def image():
    """Random 1-channel image, channels first (B, 1, H, W)."""
    return np.random.default_rng(0).random((1, 1, 200, 200)).astype(np.float32)


def test_model_mpp_and_defaults_match_nuclear_segmentation():
    """deepcell NuclearSegmentation 1.1: model_mpp 0.65, radius 10, maxima_threshold 0.1,
    interior_threshold 0.01, exclude_border False, small_objects_threshold 0."""
    assert MODEL_MPP == 0.65
    assert DEFAULT_POSTPROCESS_KWARGS == {'radius': 10, 'maxima_threshold': 0.1, 'interior_threshold': 0.01,
                                          'exclude_border': False, 'small_objects_threshold': 0}
    assert inspect.signature(DNN.predict).parameters['pad_mode'].default == 'reflect'


@pytest.mark.parametrize('mpp', [0.5, 0.65, 0.8, None])  # below, at and above the model's 0.65
def test_mask_shape_and_dtype(app, image, mpp):
    mask = app.predict(image, image_mpp=mpp, verbose=False)
    assert mask.shape == (1, 1, 200, 200)
    assert np.issubdtype(mask.dtype, np.integer)


def test_image_mpp_is_required(app, image):
    with pytest.raises(TypeError):
        app.predict(image)


@pytest.mark.parametrize('mpp', [float('nan'), 0, -0.5, 'abc'])
def test_bad_image_mpp_is_rejected(app, image, mpp):
    with pytest.raises(ValueError, match='image_mpp'):
        app.predict(image, image_mpp=mpp, verbose=False)


def test_none_mpp_means_model_resolution(app, image):
    np.testing.assert_array_equal(app.predict(image, image_mpp=None, verbose=False),
                                  app.predict(image, image_mpp=MODEL_MPP, verbose=False))


def test_rescaling_changes_model_input_size(app):
    image = np.random.default_rng(1).random((1, 1, 260, 260)).astype(np.float32)
    _, t = app.predict(image, image_mpp=0.325, return_transforms=True, verbose=False)
    assert t.shape == (1, 4, 130, 130)  # 260 * 0.325 / 0.65
    _, t = app.predict(image, image_mpp=1.3, return_transforms=True, verbose=False)
    assert t.shape == (1, 4, 520, 520)


def test_large_image_is_tiled(app):
    image = np.random.default_rng(1).random((2, 1, 600, 520)).astype(np.float32)
    mask, transforms = app.predict(image, image_mpp=MODEL_MPP, return_transforms=True, verbose=False)
    assert mask.shape == (2, 1, 600, 520)
    assert transforms.shape == (2, 4, 600, 520)


@pytest.mark.parametrize('shape', [(1, 1, 200, 600), (1, 1, 600, 200)])
def test_one_side_small_one_side_large(app, shape):
    image = np.random.default_rng(2).random(shape).astype(np.float32)
    mask, transforms = app.predict(image, image_mpp=MODEL_MPP, return_transforms=True, verbose=False)
    assert mask.shape == shape
    assert transforms.shape == (1, 4) + shape[2:]


def test_frames_are_segmented_independently(app):
    frames = np.random.default_rng(3).random((3, 1, 256, 256)).astype(np.float32)
    together = app.predict(frames, image_mpp=MODEL_MPP, verbose=False)
    alone = app.predict(frames[1:2], image_mpp=MODEL_MPP, verbose=False)
    np.testing.assert_array_equal(together[1:2], alone)


def test_input_is_not_modified(app):
    image = (np.random.default_rng(4).random((1, 1, 300, 300)) * 1000).astype(np.uint16)
    before = image.copy()
    app.predict(image, image_mpp=MODEL_MPP, verbose=False)
    app.predict(image, image_mpp=0.5, verbose=False)
    np.testing.assert_array_equal(image, before)


def test_watershed_reads_inner_and_outer_distance(app):
    """As in deepcell's NuclearSegmentation: seeds from the inner distance (channel 0),
    mask from the outer distance (channel 1); the fgbg head is not used."""
    image = np.random.default_rng(5).random((1, 1, 256, 256)).astype(np.float32)
    mask, t = app.predict(image, image_mpp=None, return_transforms=True, verbose=False)
    expected = deep_watershed(t[:, [0, 1]], **DEFAULT_POSTPROCESS_KWARGS).astype(int)
    np.testing.assert_array_equal(mask, expected)


def test_postprocess_kwargs_override_defaults(app, image, monkeypatch):
    seen = {}
    real = dnn_module.dnn_postprocess

    def spy(x, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return real(x, **kwargs)

    monkeypatch.setattr(dnn_module, 'dnn_postprocess', spy)
    app.predict(image, image_mpp=MODEL_MPP, verbose=False)
    assert seen == {**DEFAULT_POSTPROCESS_KWARGS, 'progbar': False}
    app.predict(image, image_mpp=MODEL_MPP, verbose=False, postprocess_kwargs={'radius': 3})
    assert seen == {**DEFAULT_POSTPROCESS_KWARGS, 'radius': 3, 'progbar': False}


def test_preprocess_kwargs_are_forwarded(app, image, monkeypatch):
    """Settings in ``preprocess_kwargs`` reach the preprocessing function and change the result."""
    seen = {}
    real = dnn_module.dnn_preprocess

    def spy(x, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return real(x, **kwargs)

    monkeypatch.setattr(dnn_module, 'dnn_preprocess', spy)
    _, default = app.predict(image, image_mpp=MODEL_MPP, return_transforms=True, verbose=False)
    assert seen == {}
    _, changed = app.predict(image, image_mpp=MODEL_MPP, return_transforms=True, verbose=False,
                             preprocess_kwargs={'kernel_size': 16})
    assert seen == {'kernel_size': 16}
    assert not np.array_equal(default, changed)


def test_verbose_shows_progress(app, image, capsys):
    app.predict(image, image_mpp=MODEL_MPP, verbose=False)
    assert 'Postprocessing frames' not in capsys.readouterr().err
    app.predict(image, image_mpp=MODEL_MPP, verbose=True)
    err = capsys.readouterr().err
    assert 'Postprocessing frames' in err and 'Running network' in err


def test_channels_last_input_is_rejected(app):
    """A (batch, H, W, 1) array raises an error that says to ``moveaxis``."""
    with pytest.raises(ValueError, match='moveaxis'):
        app.predict(np.zeros((1, 256, 256, 1), np.float32), image_mpp=MODEL_MPP)


@pytest.mark.parametrize('shape, hint', [((256, 256), 'np.newaxis, np.newaxis'),
                                         ((3, 256, 256), r'image\[:, np.newaxis\]'),
                                         ((1, 2, 256, 256), 'channel'),
                                         ((1, 1, 1, 256, 256), 'dimensions')])
def test_wrong_rank_or_channels_is_rejected(app, shape, hint):
    with pytest.raises(ValueError, match=hint):
        app.predict(np.zeros(shape, np.float32), image_mpp=MODEL_MPP)


# --- construction and weights ---

def test_init_skips_imagenet_download(monkeypatch):
    """DNN() builds the DynamicNuclearNet network without ImageNet weights, then loads the checkpoint."""
    built = {}

    def fake_panopticnet(**kwargs):
        built.update(kwargs)
        return StubDnnNet()

    monkeypatch.setattr(dnn_module, 'PanopticNet', fake_panopticnet)
    monkeypatch.setattr(dnn_module.torch, 'load', lambda *a, **k: StubDnnNet().state_dict())
    app = DNN(model_path='unused.pth', device='cpu')
    assert built == {**DNN_MODEL_KWARGS, 'use_imagenet': False}
    assert not app.model.training
    assert app.model_mpp == 0.65


def test_default_model_path_picks_newest(tmp_path, monkeypatch):
    for name in ('torch-dnn_2026-11-01.pth', 'torch-dnn_2026-11-20.pth', 'other.pth'):
        (tmp_path / name).touch()
    assert default_model_path(tmp_path).name == 'torch-dnn_2026-11-20.pth'
    loaded = []
    monkeypatch.setattr(dnn_module, 'DEFAULT_MODEL_DIR', tmp_path)
    monkeypatch.setattr(dnn_module, 'PanopticNet', lambda **kwargs: StubDnnNet())
    monkeypatch.setattr(dnn_module.torch, 'load', lambda path, **k: loaded.append(path) or StubDnnNet().state_dict())
    DNN(device='cpu')
    assert loaded == [tmp_path / 'torch-dnn_2026-11-20.pth']


def test_missing_default_weights_explain_how_to_train(tmp_path, monkeypatch):
    monkeypatch.setattr(dnn_module, 'DEFAULT_MODEL_DIR', tmp_path)
    with pytest.raises(FileNotFoundError, match='python -m torch_dnn.train'):
        DNN(device='cpu')


def _fresh_interpreter(code):
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_import_does_not_change_torch_threads():
    """Importing ``torch_dnn.dnn`` leaves the caller's torch thread count alone
    (the old module called ``torch.set_num_threads(24)``)."""
    n = _fresh_interpreter(
        'import torch; torch.set_num_threads(3); import torch_dnn.dnn; print(torch.get_num_threads())')
    assert n == '3'


def test_utils_import_does_not_load_matplotlib():
    loaded = _fresh_interpreter("import sys, torch_dnn.utils; print('matplotlib' in sys.modules)")
    assert loaded == 'False'


@pytest.fixture(scope='module')
def random_checkpoint(tmp_path_factory):
    """A freshly initialised DynamicNuclearNet, saved as train.py saves it (a state dict)."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp('weights') / 'torch-dnn_random.pth'
    torch.save(PanopticNet(**DNN_MODEL_KWARGS, use_imagenet=False).state_dict(), path)
    return path


@pytest.mark.slow
def test_real_network_round_trip(random_checkpoint):
    app = DNN(model_path=random_checkpoint, device='cpu')
    assert isinstance(app.model, PanopticNet) and not app.model.training
    image = np.random.default_rng(6).random((2, 1, 256, 256)).astype(np.float32)
    mask, t = app.predict(image, image_mpp=MODEL_MPP, return_transforms=True, verbose=False)
    assert mask.shape == (2, 1, 256, 256)
    assert t.shape == (2, 4, 256, 256)
    np.testing.assert_allclose(t[:, 2:4].sum(axis=1), 1, atol=1e-5)
    assert (t[:, :2] >= 0).all()


@pytest.mark.slow
@pytest.mark.skipif(not EXAMPLE_IMAGE.exists(), reason='example image not found')
def test_example_image_end_to_end(random_checkpoint):
    """The README flow: one 512x512 uint16 frame, with a random-weights network."""
    import tifffile
    frame = tifffile.imread(EXAMPLE_IMAGE)
    app = DNN(model_path=random_checkpoint, device='cpu')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = app.predict(frame[np.newaxis, np.newaxis], image_mpp=MODEL_MPP, verbose=False)
    assert mask.shape == (1, 1) + frame.shape
    assert np.issubdtype(mask.dtype, np.integer)


@pytest.mark.slow
def test_mismatched_checkpoints_raise_value_error(random_checkpoint, tmp_path):
    sd = torch.load(random_checkpoint)
    del sd['arch_version']
    torch.save(sd, tmp_path / 'old.pth')
    with pytest.raises(ValueError, match='original torch_dnn network'):
        DNN(model_path=tmp_path / 'old.pth', device='cpu')

    mesmer_like = PanopticNet(use_imagenet=False, crop_size=128, in_channels=2,
                              n_semantic_classes=[1, 3, 1, 3]).state_dict()
    torch.save(mesmer_like, tmp_path / 'mesmer.pth')
    with pytest.raises(ValueError, match='does not fit'):
        DNN(model_path=tmp_path / 'mesmer.pth', device='cpu')
