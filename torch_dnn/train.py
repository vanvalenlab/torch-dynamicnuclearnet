"""Train DynamicNuclearNet (nuclear segmentation) with deepcell's recipe.

Copied from ``torch_mesmer.train`` and adapted to deepcell's DynamicNuclearNet
training notebook (``notebooks/training/panopticnets/Nuclear Segmentation -
DeepWatershed.ipynb``):

* Keras ``Adam(lr=1e-4, clipnorm=0.001)`` (epsilon 1e-7, per-tensor clipping);
* ``LearningRateScheduler(rate_scheduler(lr, decay=0.99))``: ``lr * 0.99**epoch``;
* ``ReduceLROnPlateau(val_loss, factor=0.1, patience=5, min_delta=1e-4)``;
* ``EarlyStopping(patience=10, restore_best_weights=True)`` on ``val_loss``;
* ``ModelCheckpoint(save_best_only=True)`` on ``val_loss``;
* batch size 16, 16 epochs, ``n // batch_size`` steps per epoch.

One deliberate change from the notebook: in Keras the scheduler sets the
learning rate from the epoch number alone at the start of every epoch, so a
plateau reduction is overwritten before the next epoch trains and the plateau
callback has no effect. Here the 0.99 decay multiplies the *current* learning
rate instead, so a plateau cut persists: with no cuts the rates are identical
to the notebook's, and after each cut they are 10x lower.
"""

import os
import argparse
import datetime
import json
import random
import subprocess
import time

import numpy as np
import torch
import zarr

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from torch_dnn.loss import SemanticLoss, LossTracker
from torch_dnn.loaders import (CLAHE_KERNEL_SIZE, IN_TRANSFORMS, MIN_OBJECT_SIZE, MIN_OBJECTS,
                               TRANSFORMS_KWARGS, ZOOM_MIN, create_data_loaders)
from torch_dnn.preprocess import DEFAULT_DATA_DIR
from torch_dnn.utils import create_sample_overlay

# Optimiser settings of the DynamicNuclearNet notebook.
ADAM_EPSILON = 1e-7   # tf.keras Adam default; PyTorch's default is 1e-8
CLIPNORM = 0.001      # Keras Adam(clipnorm=0.001): applied to each gradient tensor separately
LR_DECAY = 0.99       # lr = lr0 * 0.99**epoch (deepcell rate_scheduler)
LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 16
CE_WEIGHT = 0.01      # on the fgbg (cross-entropy) head
SEED = 0
CROP_SIZE = 256
ROTATION_RANGE = 180
PLATEAU_KWARGS = {'factor': 0.1, 'patience': 5, 'min_delta': 1e-4, 'cooldown': 0, 'min_lr': 0.0}
EARLY_STOPPING_PATIENCE = 10
N_SEMANTIC_CLASSES = [1, 1, 2]   # inner distance, outer distance, fgbg

# Relative to the folder training is started from, as in torch_mesmer (``data/`` is git-ignored).
DEFAULT_MODEL_ROOT = Path('data') / 'model'   # checkpoints: data/model/<timestamp>/
DEFAULT_LOG_ROOT = Path('data') / 'logs'      # TensorBoard, run_config.json, history.jsonl


def seed_everything(seed=0):
    """Seed Python, NumPy and PyTorch (CPU and every GPU).

    DataLoader workers get their own seeds from the loader's generator (see
    ``create_data_loaders``), so this plus that generator makes a run repeatable
    on the same hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model, lr=LR):
    """Adam with the Keras epsilon, plus the per-epoch exponential LR decay.

    Call ``scheduler.step()`` once at the end of every epoch, so epoch ``e``
    (counting from 0) trains at ``lr * LR_DECAY**e`` times any reductions made
    in between. Unlike Keras's ``LearningRateScheduler`` (which ignores the
    current learning rate and so undoes ``ReduceLROnPlateau``), the decay
    multiplies the current rate, so plateau reductions persist.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=ADAM_EPSILON)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: LR_DECAY)
    return optimizer, scheduler


class ReduceLROnPlateau:
    """Keras 2.8 ``ReduceLROnPlateau`` (``mode='min'``), stepped once per epoch.

    The loss improves if it is below ``best - min_delta``. After ``patience``
    epochs in a row without improvement, every parameter group's learning
    rate is multiplied by ``factor`` (not below ``min_lr``) and the count
    restarts. ``torch.optim.lr_scheduler.ReduceLROnPlateau`` differs (it
    waits ``patience + 1`` epochs), hence this class.
    """

    def __init__(self, optimizer, factor=0.1, patience=5, min_delta=1e-4, cooldown=0, min_lr=0.0):
        if factor >= 1.0:
            raise ValueError('factor must be < 1')
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.best = np.inf
        self.wait = 0
        self.cooldown_counter = 0

    def step(self, current):
        """Record one epoch's monitored value.

        Returns:
            The new learning rate if it was lowered this epoch, else ``None``.
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            return None
        if self.cooldown_counter > 0:
            return None

        self.wait += 1
        if self.wait < self.patience:
            return None

        self.wait = 0
        new_lr = None
        for group in self.optimizer.param_groups:
            old_lr = group['lr']
            if old_lr > np.float32(self.min_lr):
                group['lr'] = max(old_lr * self.factor, self.min_lr)
                new_lr = group['lr']
        self.cooldown_counter = self.cooldown
        return new_lr


class EarlyStopping:
    """Keras 2.8 ``EarlyStopping`` (``mode='min'``, no baseline), stepped once per epoch.

    Training stops once ``patience`` epochs have passed without the loss
    dropping below the best so far (by more than ``min_delta``), but never
    after the first epoch. With ``restore_best_weights``, the model gets the
    weights of its best epoch back *when it stops early*; as in Keras 2.8, a
    run that reaches its last epoch keeps that epoch's weights.
    """

    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.restore_best_weights = restore_best_weights
        self.best = np.inf
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = None

    @staticmethod
    def _snapshot(model):
        return {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}

    def step(self, model, current, epoch):
        """Record one epoch's monitored value.

        Returns:
            ``True`` if training should stop now (the best weights have then
            been loaded into ``model`` if ``restore_best_weights``).
        """
        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = self._snapshot(model)

        self.wait += 1
        if current - self.min_delta < self.best:
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self._snapshot(model)
            self.wait = 0

        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


@torch.no_grad()
def clip_grad_norm_per_tensor_(named_parameters, max_norm=CLIPNORM):
    """Clip each parameter's gradient to norm ``max_norm`` on its own.

    This is Keras ``clipnorm`` (``tf.clip_by_norm`` on every gradient):
    ``g <- g * max_norm / max(||g||, max_norm)``. It is not
    ``torch.nn.utils.clip_grad_norm_``, which scales all gradients by one
    global norm (Keras's separate ``global_clipnorm``).

    Args:
        named_parameters: iterable of (name, parameter), e.g. ``model.named_parameters()``.
        max_norm: the per-tensor norm limit.

    Returns:
        Tensor of the gradient norms before clipping, one per parameter that has a gradient.

    Raises:
        FloatingPointError: if any gradient contains NaN or inf. The message names the
        affected parameters.
    """
    names, grads = [], []
    for name, p in named_parameters:
        if p.grad is not None:
            names.append(name)
            grads.append(p.grad)

    if not grads:
        return torch.zeros(0)

    norms = torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])

    finite = torch.isfinite(norms)
    if not finite.all():
        bad = [names[i] for i in torch.nonzero(~finite).flatten().tolist()]
        shown = ', '.join(bad[:5]) + (f', ... ({len(bad)} in total)' if len(bad) > 5 else '')
        raise FloatingPointError(
            f'Non-finite gradient in {len(bad)} of {len(names)} parameter tensors: {shown}. '
            'If the loss was finite, the problem is inside the backward pass '
            '(e.g. an overflow under mixed precision, or a bad input image).'
        )

    for g, norm in zip(grads, norms):
        g.mul_((max_norm / torch.clamp(norm, min=max_norm)).to(g.dtype))

    return norms


def train_step(model, loss_fn, optimizer, image, labels, max_norm=CLIPNORM, return_heads=False):
    """One optimisation step: forward, loss, backward, per-tensor clip, Adam update.

    Returns:
        The loss (a detached scalar tensor). With ``return_heads=True``, a tuple
        ``(loss, head_losses)``, where ``head_losses`` is a detached 1-D tensor
        of the per-head losses (``loss_fn`` must then have ``head_losses``, as
        ``SemanticLoss`` does).

    Raises:
        FloatingPointError: if the loss or any gradient is NaN or inf.
    """
    optimizer.zero_grad()

    outputs = model(image)
    if return_heads:
        heads = loss_fn.head_losses(outputs, labels)
        curr_loss = sum(heads[1:], heads[0])
    else:
        curr_loss = loss_fn(outputs, labels)

    if not torch.isfinite(curr_loss):
        raise FloatingPointError(
            f'Training loss is {curr_loss.item()}. Check the batch: images should be finite, '
            'and the fgbg targets one-hot.'
        )

    curr_loss.backward()
    clip_grad_norm_per_tensor_(model.named_parameters(), max_norm=max_norm)
    optimizer.step()

    if return_heads:
        return curr_loss.detach(), torch.stack([h.detach() for h in heads])
    return curr_loss.detach()


def head_names(n_semantic_classes):
    """Readable names for the loss of each head, for logs."""
    if list(n_semantic_classes) == N_SEMANTIC_CLASSES:
        return ['inner_distance', 'outer_distance', 'fgbg']
    return [f'head{i}' for i in range(len(n_semantic_classes))]


def git_commit():
    """The current git commit, with '-dirty' if tracked files have uncommitted changes."""
    try:
        here = Path(__file__).resolve().parent
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _train_one_epoch(model, loss_fn, optimizer, dataloader, device, epoch):
    """One pass of ``train_step`` over the training loader.

    Returns:
        ``(loss, head_losses)``: the mean loss per sample, and a list of the
        mean loss of each head.
    """
    model.train()
    tracker = LossTracker()
    head_sums = torch.zeros(len(loss_fn.n_semantic_classes), dtype=torch.float64)

    progress = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', dynamic_ncols=True, mininterval=30)
    for image, labels in progress:
        batch_size = image.shape[0]
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_loss, batch_heads = train_step(model, loss_fn, optimizer, image, labels, return_heads=True)

        tracker.update(batch_loss, batch_size=batch_size)
        head_sums += batch_heads.cpu().double() * batch_size
        progress.set_postfix({'loss': f"{tracker.get_loss():.4f}"})

    return tracker.get_loss(), (head_sums / max(tracker.total_samples, 1)).tolist()


@torch.no_grad()
def _validate(model, loss_fn, valloader, device, epoch):
    """The loss on the validation loader, with no gradient.

    The model is in eval mode, so BatchNorm uses its running statistics. In the
    semantic heads these move slowly (Keras momentum 0.99, as in deepcell), so
    for the first ~100 training steps they still describe the untrained network
    and the val loss (mostly the inner-distance head) can be 10-100x the train
    loss. That is expected and gone by the end of the first full epoch; the
    TF recipe behaves the same way.

    Returns:
        ``(loss, head_losses, sample)``. ``loss`` and ``head_losses`` are as in
        ``_train_one_epoch``. ``sample`` is ``(targets, outputs)`` of the first
        image of the last batch, each ``(channels, H, W)``, for the TensorBoard
        figures.

    Raises:
        ValueError: the loader has no batches.
    """
    model.eval()
    tracker = LossTracker()
    head_sums = torch.zeros(len(loss_fn.n_semantic_classes), dtype=torch.float64)
    sample = None

    progress = tqdm(valloader, desc=f'Epoch {epoch} [Val]', dynamic_ncols=True, mininterval=30)
    for image, labels in progress:
        batch_size = image.shape[0]
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(image)
        batch_heads = torch.stack(loss_fn.head_losses(outputs, labels))

        tracker.update(batch_heads.sum(), batch_size=batch_size)
        head_sums += batch_heads.cpu().double() * batch_size
        progress.set_postfix({'loss': f"{tracker.get_loss():.4f}"})
        sample = (labels[0], outputs[0])

    if sample is None:
        raise ValueError('The validation loader has no batches (fewer images than one batch?).')
    return tracker.get_loss(), (head_sums / max(tracker.total_samples, 1)).tolist(), sample


def _log_to_tensorboard(writer, epoch, names, val_loss, train_heads, val_heads, sample):
    """Write the epoch's val loss, the loss of each head and the sample figure."""
    writer.add_scalar('avg_loss/val', val_loss, epoch)
    for name, train_head, val_head in zip(names, train_heads, val_heads):
        writer.add_scalar(f'head_loss_train/{name}', train_head, epoch)
        writer.add_scalar(f'head_loss_val/{name}', val_head, epoch)

    targets, outputs = sample
    writer.add_figure('sample', create_sample_overlay(targets, outputs), epoch)
    writer.flush()


def _max_gpu_memory_gib(device):
    """Peak GPU memory allocated on ``device`` so far, in GiB; ``None`` off the GPU."""
    if str(device).startswith('cuda'):
        return round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    return None


def _append_history(history_path, record):
    """Append ``record`` to ``history_path`` as one line of JSON."""
    with open(history_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def _print_epoch(names, train_loss, train_heads, val_loss, val_heads, t_train, t_epoch):
    """Print the epoch's losses (total and per head) and timings."""
    print(f'Training loss: {train_loss:.4f}  '
          + '  '.join(f'{n}={v:.4f}' for n, v in zip(names, train_heads)))
    print(f'Validation loss: {val_loss:.4f}  '
          + '  '.join(f'{n}={v:.4f}' for n, v in zip(names, val_heads)))
    print(f'Epoch time: {t_epoch:.0f} s (train {t_train:.0f} s)', flush=True)
    print()


def _save_weights(model, save_path_prefix, filename):
    """Save the model's state_dict as ``<save_path_prefix>/<filename>``."""
    torch.save(model.state_dict(), str(save_path_prefix) + "/" + filename)


def train_torch(
        dataloader,
        valloader,
        model=None,
        lr=LR,
        epochs=EPOCHS,
        save_path_prefix="data/model",
        writer=None,
        write=True,
        device='cuda',
        ce_weight=CE_WEIGHT,
        history_path=None,
        plateau_kwargs=PLATEAU_KWARGS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    ):
    """Train with the notebook's recipe; keep the checkpoint with the lowest val loss.

    Each epoch trains on every batch of ``dataloader`` and measures the loss on
    ``valloader``. Then, in the order of the notebook's callbacks: the
    checkpoint is saved if the val loss is the lowest so far
    (``ModelCheckpoint``), ``ReduceLROnPlateau`` and ``EarlyStopping`` see the
    val loss, and the learning rate for the next epoch is multiplied by 0.99
    (so a plateau reduction carries over; see the module docstring).

    Args:
        dataloader, valloader: loaders of ``(images, targets)`` batches, as
            ``create_data_loaders`` returns.
        model: the network to train (required), already on ``device``. It
            must have ``n_semantic_classes``, as ``PanopticNet`` does.
        lr: initial learning rate.
        epochs: maximum number of epochs.
        save_path_prefix: folder for the checkpoints.
        writer: optional TensorBoard ``SummaryWriter``. Gets the lr, the losses
            and a sample figure every epoch.
        write: if ``False``, save no checkpoints.
        device: where to move each batch.
        ce_weight: weight of the cross-entropy head (``SemanticLoss``).
        history_path: optional JSON-lines file, see below.
        plateau_kwargs: ``ReduceLROnPlateau`` settings; ``None`` leaves it out.
        early_stopping_patience: ``EarlyStopping`` patience; ``None`` disables it.

    Returns:
        ``model``, trained in place.

    Side effects:
        Saves the state_dict as ``saved_model_best_dict.pth`` (whenever the val
        loss is the lowest so far) and ``last_model_dict.pth`` (at the end; after
        an early stop, these are the best epoch's weights) under
        ``save_path_prefix``. If ``history_path`` is given, appends one JSON
        line per epoch to it: lr, train and val loss (total and per head),
        whether it was the best so far, the plateau and early-stopping events,
        and timings. Prints a summary of each epoch.
    """
    assert model is not None, "Please specify a model"

    names = head_names(model.n_semantic_classes)
    loss_fn = SemanticLoss(n_semantic_classes=model.n_semantic_classes, ce_weight=ce_weight)
    optimizer, scheduler = make_optimizer(model, lr=lr)
    plateau = ReduceLROnPlateau(optimizer, **plateau_kwargs) if plateau_kwargs is not None else None
    stopper = EarlyStopping(early_stopping_patience) if early_stopping_patience is not None else None
    best_val_loss = None

    for epoch in range(epochs):
        epoch_lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar('lr', epoch_lr, epoch)

        t_start = time.time()
        train_loss, train_heads = _train_one_epoch(model, loss_fn, optimizer, dataloader, device, epoch)
        t_train = time.time() - t_start
        if writer is not None:
            writer.add_scalar('avg_loss/train', train_loss, epoch)

        val_loss, val_heads, sample = _validate(model, loss_fn, valloader, device, epoch)
        t_epoch = time.time() - t_start

        if writer is not None:
            _log_to_tensorboard(writer, epoch, names, val_loss, train_heads, val_heads, sample)

        # ModelCheckpoint(save_best_only=True)
        is_best = best_val_loss is None or val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            if write:
                _save_weights(model, save_path_prefix, "saved_model_best_dict.pth")
            print("\nNew best model.\n")

        # ReduceLROnPlateau, then EarlyStopping (which may restore the best weights).
        plateau_lr = plateau.step(val_loss) if plateau is not None else None
        if plateau_lr is not None:
            print(f'ReduceLROnPlateau: lr -> {plateau_lr:.3g}')
        stop = stopper.step(model, val_loss, epoch) if stopper is not None else False

        # Per-epoch decay: lr for the next epoch = current lr * 0.99 (keeps plateau cuts).
        scheduler.step()

        if history_path is not None:
            _append_history(history_path, {
                'epoch': epoch,
                'lr': epoch_lr,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_head_loss': dict(zip(names, train_heads)),
                'val_head_loss': dict(zip(names, val_heads)),
                'is_best': is_best,
                'plateau_lr': plateau_lr,
                'early_stop': stop,
                'train_seconds': round(t_train, 1),
                'epoch_seconds': round(t_epoch, 1),
                'max_gpu_memory_gib': _max_gpu_memory_gib(device),
            })

        _print_epoch(names, train_loss, train_heads, val_loss, val_heads, t_train, t_epoch)

        if stop:
            print(f'Early stopping after epoch {epoch}: no improvement for {stopper.patience} epochs. '
                  f'Restored the weights of epoch {stopper.best_epoch}.')
            break

    if write:
        _save_weights(model, save_path_prefix, "last_model_dict.pth")

    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train torch_dnn with deepcell's DynamicNuclearNet recipe.")
    p.add_argument('--device', default='cuda:0',
                   help="torch device. Pick the physical GPU with CUDA_VISIBLE_DEVICES; then it is 'cuda:0'.")
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--num-workers', type=int, default=4, help='DataLoader worker processes (shared server: keep small).')
    p.add_argument('--threads', type=int, default=4, help='torch intra- and inter-op threads in the main process.')
    p.add_argument('--data-path', default=str(DEFAULT_DATA_DIR),
                   help='folder with train.zarr and val.zarr (made by torch_dnn.preprocess)')
    p.add_argument('--model-path', default=str(DEFAULT_MODEL_ROOT),
                   help='checkpoints go in <model-path>/<timestamp>/ (default: data/model in the current folder)')
    p.add_argument('--run-info', default=str(DEFAULT_LOG_ROOT),
                   help='TensorBoard run, run_config.json and history.jsonl go in <run-info>/<timestamp>/ '
                        '(default: data/logs in the current folder)')
    p.add_argument('--no-write', action='store_true', help="don't save checkpoints")
    p.add_argument('--max-train-images', type=int, default=None,
                   help='use only the first N training images (quick tests; loads them into memory)')
    p.add_argument('--max-val-images', type=int, default=None,
                   help='use only the first N validation images (quick tests; loads them into memory)')
    p.add_argument('--notes', default='', help='free text stored in run_config.json')
    return p.parse_args(argv)


def _model_kwargs():
    """``torch_dnn.model.DNN_MODEL_KWARGS``: the DynamicNuclearNet architecture."""
    from torch_dnn.model import DNN_MODEL_KWARGS
    return dict(DNN_MODEL_KWARGS)


def _training_config(args):
    """The settings of a training run: the notebook's recipe plus the command-line options."""
    model_kwargs = _model_kwargs()
    return {
        'model_path': args.model_path,
        'data_path': Path(args.data_path),
        'run_info': args.run_info,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': LR,
        'adam_epsilon': ADAM_EPSILON,
        'clipnorm': CLIPNORM,
        'lr_decay': LR_DECAY,
        'plateau': PLATEAU_KWARGS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'model_kwargs': model_kwargs,
        # Data pipeline of the notebook; passed to create_data_loaders.
        'crop_size': CROP_SIZE,
        'zoom_min': ZOOM_MIN,
        'rotation_range': ROTATION_RANGE,
        'in_transforms': list(IN_TRANSFORMS),
        'transforms_kwargs': TRANSFORMS_KWARGS,
        'kernel_size': CLAHE_KERNEL_SIZE,
        'min_object_size': MIN_OBJECT_SIZE,
        'min_objects': MIN_OBJECTS,           # train and val
        'max_train_images': args.max_train_images,
        'max_val_images': args.max_val_images,
        'num_workers': args.num_workers,
        'write': not args.no_write,
        'device': args.device,
        'n_semantic_classes': list(model_kwargs.get('n_semantic_classes', N_SEMANTIC_CLASSES)),
        'ce_weight': CE_WEIGHT,
        'seed': SEED,
        'threads': args.threads,
        'notes': args.notes,
    }


def _make_run_dirs(config, start_time):
    """Create ``<run_info>/<start_time>`` and, if checkpoints are saved,
    ``<model_path>/<start_time>``. (With the defaults both are the same folder.)

    Returns:
        ``(run_info, model_path)``, the two folders as strings.
    """
    run_info = str(config['run_info']) + '/' + start_time
    model_path = str(config['model_path']) + '/' + start_time

    if not os.path.isdir(run_info):
        os.makedirs(run_info, exist_ok=True)
    if not os.path.isdir(model_path) and config['write']:
        os.makedirs(model_path, exist_ok=True)

    return run_info, model_path


def _record_run(config, start_time, run_info, model_path):
    """Write ``run_config.json`` to ``run_info``: the settings, the git commit and the environment.

    If tracked files have uncommitted changes, also write them to
    ``uncommitted.diff``, so the run can be reproduced from the commit plus
    that patch.
    """
    run_record = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()},
        'git_commit': git_commit(),
        'start_time': start_time,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_version': torch.__version__,
        'model_dir': model_path,
    }
    with open(f'{run_info}/run_config.json', 'w') as f:
        json.dump(run_record, f, indent=2, default=str)
    if run_record['git_commit'].endswith('-dirty'):
        try:
            diff = subprocess.run(['git', 'diff', 'HEAD'], cwd=Path(__file__).resolve().parent,
                                  capture_output=True, text=True, check=True).stdout
            with open(f'{run_info}/uncommitted.diff', 'w') as f:
                f.write(diff)
        except (OSError, subprocess.CalledProcessError):
            pass
    print(f"Run {start_time}: checkpoints in {model_path}, logs in {run_info}, commit {run_record['git_commit']}")


def _build_model(config):
    """A new ``PanopticNet(**DNN_MODEL_KWARGS)``, on ``config['device']``."""
    from torch_dnn.model import PanopticNet

    print("Initializing model:")
    print()

    model = PanopticNet(**config['model_kwargs'])
    model = model.to(config['device'])

    print("Panoptic Model:")
    print(f"    Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    return model


def _build_loaders(config, z_train, z_val, semantic_heads=None):
    """The train and val loaders for ``config`` (see ``create_data_loaders``)."""
    return create_data_loaders(
        z_train,
        z_val,
        crop_size=config['crop_size'],
        zoom_min=config['zoom_min'],
        rotation_range=config['rotation_range'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        in_transforms=config['in_transforms'],
        transforms_kwargs=config['transforms_kwargs'],
        kernel_size=config['kernel_size'],
        min_object_size=config['min_object_size'],
        min_objects=config['min_objects'],
        semantic_heads=semantic_heads,
        seed=config['seed'],
        max_train_images=config['max_train_images'],
        max_val_images=config['max_val_images'],
    )


def main(argv=None):
    """Train DynamicNuclearNet with the notebook's recipe.

    Reads ``train.zarr`` and ``val.zarr`` from ``--data-path`` (default
    ``~/.deepcell/dynamic_nuclear_net/``). Each run gets a timestamped folder
    under ``--run-info`` (TensorBoard logs, ``run_config.json``,
    ``history.jsonl``) and one under ``--model-path`` (the checkpoints, see
    ``train_torch``); by default ``data/logs/<timestamp>/`` and
    ``data/model/<timestamp>/`` in the current folder. At the end it prints
    how to make the best checkpoint the app's default model.

    Args:
        argv: command-line arguments (default: ``sys.argv[1:]``); see ``parse_args``.
    """
    args = parse_args(argv)

    # Set here rather than at import time, so importing train.py (from tests or
    # other scripts) leaves the thread pool alone. Keep it small on a shared machine.
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:  # already set in this process (e.g. a second main() call in tests)
        pass

    config = _training_config(args)
    seed_everything(config['seed'])
    start_time = f"{datetime.datetime.now():%Y%m%d%H%M%S}"

    z_train = zarr.open(f"{config['data_path']}/train.zarr", mode="r")
    z_val = zarr.open(f"{config['data_path']}/val.zarr", mode="r")

    run_info, model_path = _make_run_dirs(config, start_time)
    writer = SummaryWriter(run_info)
    _record_run(config, start_time, run_info, model_path)

    model = _build_model(config)
    train_data, val_data = _build_loaders(config, z_train, z_val, semantic_heads=model.n_semantic_classes)
    print(f'{len(train_data.dataset)} training images ({len(train_data)} steps per epoch), '
          f'{len(val_data.dataset)} validation images ({len(val_data)} steps).', flush=True)

    train_torch(
        train_data,
        val_data,
        model=model,
        lr=config['lr'],
        epochs=config['epochs'],
        save_path_prefix=model_path,
        writer=writer,
        write=config['write'],
        ce_weight=config['ce_weight'],
        device=config['device'],
        history_path=f'{run_info}/history.jsonl',
    )

    writer.close()
    if config['write']:
        _print_install_hint(model_path, start_time)


def _print_install_hint(model_path, start_time):
    """Tell the user how to make this run's best checkpoint the app's default model."""
    from torch_dnn.dnn import DEFAULT_MODEL_DIR
    best = Path(model_path).resolve() / 'saved_model_best_dict.pth'
    date = f'{start_time[:4]}-{start_time[4:6]}-{start_time[6:8]}'
    print(f'\nBest weights: {best}\n'
          f'Evaluate them with:  python -m torch_dnn.eval --model-path {best}\n'
          f'To make them the default for DNN() (newest torch-dnn*.pth in {DEFAULT_MODEL_DIR}):\n'
          f'    cp {best} {DEFAULT_MODEL_DIR}/torch-dnn_{date}.pth')

if __name__ == "__main__":
    main()
