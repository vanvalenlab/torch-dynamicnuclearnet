"""Tests for the optimiser setup and training loop in ``torch_dnn.train`` and
the loader settings in ``torch_dnn.loaders`` against deepcell's
DynamicNuclearNet notebook: Keras ``Adam(lr=1e-4, clipnorm=0.001)`` with
ε = 1e-7, ``LearningRateScheduler(rate_scheduler(lr, decay=0.99))``,
``ReduceLROnPlateau(factor=0.1, patience=5, min_delta=1e-4)``,
``EarlyStopping(patience=10, restore_best_weights=True)``,
``steps_per_epoch = n // batch_size`` and seed 0 (TF 2.8 / Keras 2.8 callbacks).

Copied from torch-mesmer's ``tests/test_train.py``, plus the callbacks the
Mesmer recipe does not use and a smoke test of the whole ``main()``.

The clipping reference is ``tf.clip_by_norm`` (what Keras ``clipnorm`` applies
to every gradient), transcribed in NumPy: ``t · c / max(‖t‖₂, c)``.
"""

import json

import numpy as np
import pytest
import torch
from torch import nn

from torch_dnn import train as tr
from torch_dnn.loaders import create_data_loaders
from torch_dnn.loss import SemanticLoss
from torch_dnn.train import (ADAM_EPSILON, CLIPNORM, EarlyStopping, ReduceLROnPlateau,
                             clip_grad_norm_per_tensor_, make_optimizer, train_step, train_torch)

from tests.conftest import random_blobs


def ref_clip_by_norm(t, clip_norm):
    norm = np.sqrt((t.astype(np.float64) ** 2).sum())
    return t * clip_norm / max(norm, clip_norm)


def _model_with_grads(rng, scales):
    """A model whose parameters have gradients with norms of roughly ``scales``."""
    model = nn.Sequential(*[nn.Linear(4, 4, bias=False) for _ in scales])
    for p, s in zip(model.parameters(), scales):
        p.grad = torch.from_numpy(rng.normal(size=tuple(p.shape)).astype(np.float32))
        p.grad *= s / p.grad.norm()
    return model


# ---------------------------------------------------------------------------
# Per-tensor gradient clipping
# ---------------------------------------------------------------------------

def test_clip_matches_keras_clip_by_norm():
    rng = np.random.default_rng(0)
    model = _model_with_grads(rng, [1e-5, 5e-4, 1e-3, 0.02, 3.0])
    before = [p.grad.numpy().copy() for p in model.parameters()]
    norms = clip_grad_norm_per_tensor_(model.named_parameters(), max_norm=CLIPNORM)
    for g0, p, n in zip(before, model.parameters(), norms):
        np.testing.assert_allclose(p.grad.numpy(), ref_clip_by_norm(g0, CLIPNORM), rtol=1e-6)
        assert n.item() == pytest.approx(np.linalg.norm(g0), rel=1e-5)


def test_clip_is_per_tensor_not_global():
    """A large gradient on one tensor must not shrink a small one on another."""
    rng = np.random.default_rng(1)
    model = _model_with_grads(rng, [1e-4, 10.0])
    small_before = next(model.parameters()).grad.clone()
    clip_grad_norm_per_tensor_(model.named_parameters())
    small, large = (p.grad for p in model.parameters())
    torch.testing.assert_close(small, small_before)
    assert large.norm().item() == pytest.approx(CLIPNORM, rel=1e-5)


def test_clip_leaves_zero_gradient_alone():
    model = nn.Linear(3, 3)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    clip_grad_norm_per_tensor_(model.named_parameters())
    assert all(torch.isfinite(p.grad).all() and (p.grad == 0).all() for p in model.parameters())


def test_clip_skips_parameters_without_gradients():
    model = nn.Linear(3, 3)
    model.weight.grad = torch.ones_like(model.weight)
    norms = clip_grad_norm_per_tensor_(model.named_parameters())
    assert len(norms) == 1
    assert model.bias.grad is None


def test_clip_names_the_non_finite_parameter():
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    model[1].bias.grad[0] = float('nan')
    with pytest.raises(FloatingPointError, match=r'1 of 4 .*1\.bias'):
        clip_grad_norm_per_tensor_(model.named_parameters())


# ---------------------------------------------------------------------------
# Adam ε and the LR schedule
# ---------------------------------------------------------------------------

def test_adam_uses_keras_epsilon_and_lr():
    optimizer, _ = make_optimizer(nn.Linear(2, 2), lr=1e-4)
    assert isinstance(optimizer, torch.optim.Adam)
    group = optimizer.param_groups[0]
    assert group['eps'] == ADAM_EPSILON == 1e-7
    assert group['lr'] == pytest.approx(1e-4)
    assert group['betas'] == (0.9, 0.999)
    assert group['weight_decay'] == 0


def test_lr_decays_by_0p99_per_epoch(ref):
    optimizer, scheduler = make_optimizer(nn.Linear(2, 2), lr=1e-4)
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    schedule = ref.generators.rate_scheduler(lr=1e-4, decay=0.99)
    np.testing.assert_allclose(lrs, [schedule(e) for e in range(100)], rtol=1e-12)


def test_recipe_constants():
    assert tr.LR == 1e-4 and tr.BATCH_SIZE == 16 and tr.EPOCHS == 16 and tr.SEED == 0
    assert tr.LR_DECAY == 0.99 and tr.CE_WEIGHT == 0.01 and tr.CROP_SIZE == 256
    assert tr.PLATEAU_KWARGS == {'factor': 0.1, 'patience': 5, 'min_delta': 1e-4, 'cooldown': 0, 'min_lr': 0.0}
    assert tr.EARLY_STOPPING_PATIENCE == 10


# ---------------------------------------------------------------------------
# Keras callbacks: ReduceLROnPlateau, EarlyStopping
# ---------------------------------------------------------------------------

def _keras_plateau_lrs(losses, lr, factor=0.1, patience=5, min_delta=1e-4):
    """Keras 2.8 ReduceLROnPlateau.on_epoch_end (cooldown 0, min_lr 0), transcribed:
    the lr after each epoch's callback."""
    best, wait, out = np.inf, 0, []
    for current in losses:
        if current < best - min_delta:
            best, wait = current, 0
        else:
            wait += 1
            if wait >= patience:
                lr, wait = lr * factor, 0
        out.append(lr)
    return out


@pytest.mark.parametrize('losses', [
    [1.0] * 13,
    [1.0, 0.9, 0.95, 0.9, 0.89995, 0.91, 0.92, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4],
    list(np.linspace(1, 0.5, 12)),
])
def test_plateau_matches_keras(losses):
    optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1.0)
    plateau = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_delta=1e-4)
    lrs = []
    for v in losses:
        plateau.step(v)
        lrs.append(optimizer.param_groups[0]['lr'])
    np.testing.assert_allclose(lrs, _keras_plateau_lrs(losses, 1.0))


def test_plateau_waits_patience_epochs_not_patience_plus_one():
    optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1.0)
    plateau = ReduceLROnPlateau(optimizer, patience=5)
    reductions = [plateau.step(1.0) for _ in range(6)]
    assert reductions[:5] == [None] * 5 and reductions[5] == pytest.approx(0.1)


def test_plateau_reduction_persists_through_the_decay():
    """Deliberate change from Keras: the 0.99 decay multiplies the current lr, so a
    plateau cut is kept (in Keras the schedule would reset it the next epoch)."""
    optimizer, scheduler = make_optimizer(nn.Linear(2, 2), lr=1e-4)
    plateau = ReduceLROnPlateau(optimizer, patience=2)
    lrs = []
    for loss in [1.0, 1.0, 1.0, 1.0, 1.0, 0.5]:
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        plateau.step(loss)  # cuts after epochs 2 and 4 (no improvement for 2 epochs)
        scheduler.step()
    cuts = np.array([1, 1, 1, 0.1, 0.1, 0.01])
    np.testing.assert_allclose(lrs, 1e-4 * 0.99 ** np.arange(6) * cuts, rtol=1e-12)


def _keras_early_stop_epoch(losses, patience):
    """Keras 2.8 EarlyStopping.on_epoch_end (min_delta 0, no baseline), transcribed:
    (epoch it stops at or None, best epoch)."""
    best, wait, best_epoch = np.inf, 0, 0
    for epoch, current in enumerate(losses):
        wait += 1
        if current < best:
            best, best_epoch, wait = current, epoch, 0
        if wait >= patience and epoch > 0:
            return epoch, best_epoch
    return None, best_epoch


@pytest.mark.parametrize('losses, patience', [
    ([1.0, 2.0, 2.0, 2.0, 2.0], 3),
    ([1.0, 0.5, 0.6, 0.4, 0.45, 0.45, 0.45], 3),
    ([1.0, 0.9, 0.8, 0.7], 2),
    ([1.0, 1.0, 1.0], 1),
    ([2.0] + [1.0] * 12, 10),
])
def test_early_stopping_matches_keras_and_restores_best(losses, patience):
    model = nn.Linear(2, 2, bias=False)
    stopper = EarlyStopping(patience=patience)
    stopped = None
    for epoch, v in enumerate(losses):
        with torch.no_grad():
            model.weight.fill_(epoch)   # weights record the epoch they come from
        if stopper.step(model, v, epoch):
            stopped = epoch
            break
    expected_stop, expected_best = _keras_early_stop_epoch(losses, patience)
    assert stopped == expected_stop
    if stopped is not None:
        assert stopper.best_epoch == expected_best
        assert (model.weight == expected_best).all()


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------

class _TinyHeads(nn.Module):
    """Outputs the DynamicNuclearNet [1, 1, 2] head layout from a 1-channel image:
    two ReLU channels and one 2-class softmax."""

    n_semantic_classes = [1, 1, 2]

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 1)
        self.register_buffer('arch_version', torch.tensor(1))

    def forward(self, x):
        y = self.conv(x)
        return torch.cat([torch.relu(y[:, :2]), torch.softmax(y[:, 2:], dim=1)], dim=1)


def _tiny_batch(rng):
    x = torch.from_numpy(rng.normal(size=(2, 1, 16, 16)).astype(np.float32))
    classes = rng.integers(0, 2, size=(2, 16, 16))
    y = np.concatenate([rng.random((2, 2, 16, 16)), np.moveaxis(np.eye(2)[classes], -1, 1)], 1)
    return x, torch.from_numpy(y.astype(np.float32))


def test_train_step_reduces_loss():
    torch.manual_seed(0)
    model = _TinyHeads()
    x, y = _tiny_batch(np.random.default_rng(2))
    loss_fn = SemanticLoss([1, 1, 2])
    optimizer, _ = make_optimizer(model, lr=1e-2)
    first = train_step(model, loss_fn, optimizer, x, y)
    for _ in range(50):
        last = train_step(model, loss_fn, optimizer, x, y)
    assert not first.requires_grad
    assert last < first


def test_train_step_return_heads_matches_total():
    """With ``return_heads=True`` the step gives the same total loss, plus the per-head parts."""
    x, y = _tiny_batch(np.random.default_rng(4))
    results = []
    for return_heads in (False, True):
        torch.manual_seed(0)
        model = _TinyHeads()
        optimizer, _ = make_optimizer(model)
        results.append(train_step(model, SemanticLoss([1, 1, 2]), optimizer, x, y, return_heads=return_heads))
    total, (total_h, heads) = results
    assert heads.shape == (3,) and not heads.requires_grad
    torch.testing.assert_close(total_h, total)
    torch.testing.assert_close(heads.sum(), total)


def test_train_step_rejects_non_finite_loss():
    model = _TinyHeads()
    x, y = _tiny_batch(np.random.default_rng(3))
    x[0, 0, 0, 0] = float('nan')
    optimizer, _ = make_optimizer(model)
    with pytest.raises(FloatingPointError, match='Training loss is nan'):
        train_step(model, SemanticLoss([1, 1, 2]), optimizer, x, y)


# ---------------------------------------------------------------------------
# Loader: drop_last and seeding
# ---------------------------------------------------------------------------

def _fake_split(n):
    # No 'meta': the loader does not read the pixel size.
    return {'X': np.zeros((n, 1, 8, 8), np.float32), 'y': np.zeros((n, 1, 8, 8), np.int32)}


def test_loaders_drop_the_partial_batch():
    train, val = create_data_loaders(_fake_split(21), _fake_split(13), batch_size=8, num_workers=0,
                                     crop_size=8, min_objects=0)
    assert train.drop_last and val.drop_last
    assert len(train) == 21 // 8
    assert len(val) == 13 // 8


def test_train_shuffle_is_seeded():
    def order(seed):
        train, _ = create_data_loaders(_fake_split(40), None, batch_size=8, num_workers=0, seed=seed,
                                       crop_size=8, min_objects=0)
        return [list(b) for b in train.batch_sampler]

    assert order(0) == order(0)
    assert order(0) != order(1)


# ---------------------------------------------------------------------------
# The training loop and main()
# ---------------------------------------------------------------------------

def _blob_split(n, size=64, seed=0):
    y = np.stack([random_blobs((size, size), n_cells=8, radius=(3, 7), seed=seed + i) for i in range(n)])[:, None]
    X = (y > 0).astype(np.float32) * 50 + np.random.default_rng(seed).random(y.shape, np.float32)
    return {'X': X, 'y': y.astype(np.int32)}


def test_train_torch_writes_checkpoints_and_history(tmp_path):
    torch.manual_seed(0)
    train, val = create_data_loaders(_blob_split(8), _blob_split(4, seed=50), crop_size=32, batch_size=4,
                                     num_workers=0, semantic_heads=[1, 1, 2])
    model = _TinyHeads()
    history = tmp_path / 'history.jsonl'
    train_torch(train, val, model=model, epochs=3, save_path_prefix=str(tmp_path), device='cpu',
                history_path=str(history))
    records = [json.loads(line) for line in history.read_text().splitlines()]
    assert [r['epoch'] for r in records] == [0, 1, 2]
    np.testing.assert_allclose([r['lr'] for r in records], 1e-4 * 0.99 ** np.arange(3))
    assert set(records[0]['val_head_loss']) == {'inner_distance', 'outer_distance', 'fgbg'}
    assert records[0]['is_best'] and not any(r['early_stop'] for r in records)
    for name in ('saved_model_best_dict.pth', 'last_model_dict.pth'):
        sd = torch.load(tmp_path / name)
        assert set(sd) == set(model.state_dict())   # a plain state_dict, buffers (arch_version) included


def test_train_torch_stops_early_and_restores(tmp_path, monkeypatch):
    """With patience 1 and a val loss that never improves after epoch 0, training stops
    after epoch 1 and the last checkpoint holds the epoch-0 weights."""
    losses = iter([1.0, 2.0, 3.0, 4.0])
    real_validate = tr._validate

    def fake_validate(*args, **kwargs):
        _, heads, sample = real_validate(*args, **kwargs)
        return next(losses), heads, sample

    monkeypatch.setattr(tr, '_validate', fake_validate)
    train, val = create_data_loaders(_blob_split(4), _blob_split(4, seed=9), crop_size=32, batch_size=4,
                                     num_workers=0)
    model = _TinyHeads()
    train_torch(train, val, model=model, epochs=4, save_path_prefix=str(tmp_path), device='cpu',
                early_stopping_patience=1, lr=1e-2)
    best = torch.load(tmp_path / 'saved_model_best_dict.pth')
    last = torch.load(tmp_path / 'last_model_dict.pth')
    for k in best:
        torch.testing.assert_close(best[k], last[k])
        torch.testing.assert_close(model.state_dict()[k], best[k])


def _write_zarr_split(path, split):
    import zarr
    g = zarr.open_group(str(path), mode='w')
    for name in ('X', 'y'):
        create = getattr(g, 'create_array', None) or g.create_dataset
        arr = create(name, shape=split[name].shape, chunks=(1,) + split[name].shape[1:], dtype=split[name].dtype)
        arr[:] = split[name]


def test_default_output_folders_and_install_hint(capsys):
    """Runs go to data/model and data/logs in the current folder (as in torch_mesmer), and the
    printed copy command names a file the app's default-model lookup finds."""
    from fnmatch import fnmatch
    from torch_dnn.dnn import DEFAULT_MODEL_DIR, DEFAULT_MODEL_GLOB

    args = tr.parse_args([])
    assert (args.model_path, args.run_info) == ('data/model', 'data/logs')

    tr._print_install_hint('data/model/20261120093000', '20261120093000')
    target = capsys.readouterr().out.strip().splitlines()[-1].split()[-1]
    assert target == f'{DEFAULT_MODEL_DIR}/torch-dnn_2026-11-20.pth'
    assert fnmatch(target.rsplit('/', 1)[1], DEFAULT_MODEL_GLOB)


def test_main_runs_and_records_the_run(tmp_path, monkeypatch):
    """``main()`` end to end on a tiny zarr dataset with a tiny model."""
    data = tmp_path / 'data'
    _write_zarr_split(data / 'train.zarr', _blob_split(10, size=96))
    _write_zarr_split(data / 'val.zarr', _blob_split(6, size=96, seed=40))
    monkeypatch.setattr(tr, '_model_kwargs', lambda: {'n_semantic_classes': [1, 1, 2], 'tiny': True})
    monkeypatch.setattr(tr, '_build_model', lambda config: _TinyHeads())

    runs = tmp_path / 'runs'
    tr.main(['--device', 'cpu', '--epochs', '2', '--batch-size', '4', '--num-workers', '0', '--threads', '2',
             '--data-path', str(data), '--model-path', str(runs), '--run-info', str(runs),
             '--max-train-images', '9', '--notes', 'unit test'])

    (run,) = list(runs.iterdir())
    config = json.loads((run / 'run_config.json').read_text())
    assert config['batch_size'] == 4 and config['epochs'] == 2 and config['notes'] == 'unit test'
    assert config['model_kwargs'] == {'n_semantic_classes': [1, 1, 2], 'tiny': True}
    assert config['in_transforms'] == ['inner-distance', 'outer-distance', 'fgbg']
    assert config['max_train_images'] == 9 and config['git_commit']
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    assert len(history) == 2
    assert (run / 'saved_model_best_dict.pth').exists() and (run / 'last_model_dict.pth').exists()
    assert any(p.name.startswith('events.out.tfevents') for p in run.iterdir())
