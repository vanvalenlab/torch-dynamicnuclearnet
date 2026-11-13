# Training

`python -m torch_dnn.train` trains DynamicNuclearNet from scratch on the
DynamicNuclearNet segmentation dataset, with the recipe of deepcell's training
notebook (`notebooks/training/panopticnets/Nuclear Segmentation - DeepWatershed.ipynb`
in deepcell-tf).

## What you need

1. The dataset, converted to Zarr (once):

   ```bash
   nice -n 10 python -m torch_dnn.preprocess
   ```

   This reads `train.npz`, `val.npz` and `test.npz` from `~/.deepcell/dnn/` (unzipped
   from `DynamicNuclearNet-segmentation-v1_0.zip`, the v1.0 release) and writes
   `~/.deepcell/dynamic_nuclear_net/{train,val,test}.zarr`. It takes a few minutes and
   never holds a whole split in memory.

2. One GPU with about 20 GB free. A full run uses about 17 GB on the card (15.5 GB
   for the model and batches; batch size 16).

## Start a run

On the shared lab server, use one of the GPUs you are allowed to use, lower the
priority, and run inside a named `tmux` session so the run survives a dropped
connection:

```bash
tmux new -s <yourname>_dnn_train
export CUDA_VISIBLE_DEVICES=4                      # the physical GPU; inside Python it is then cuda:0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nice -n 10 python -m torch_dnn.train --notes "first full run"
```

(Detach with `Ctrl-b d`; come back with `tmux attach -t <yourname>_dnn_train`.)

**How long:** on an RTX 6000 Ada, one epoch is about 100 seconds (80 s of training on
4950 images in 309 batches, 20 s of validation on 1417 images), so the default 16 epochs
take about 30 minutes.

### Options

| Option | Default | What it does |
|---|---|---|
| `--epochs` | 16 | Maximum number of passes over the training images. |
| `--batch-size` | 16 | Images per training step. Smaller uses less GPU memory but changes the recipe. |
| `--num-workers` | 4 | Background processes that prepare images. Keep it small on a shared machine. |
| `--threads` | 4 | CPU threads PyTorch itself uses. |
| `--device` | `cuda:0` | Which device to train on (after `CUDA_VISIBLE_DEVICES`). |
| `--data-path` | `~/.deepcell/dynamic_nuclear_net` | Folder holding `train.zarr` and `val.zarr`. |
| `--model-path` | `data/model` | Weights go in `<model-path>/<timestamp>/`. |
| `--run-info` | `data/logs` | Logs go in `<run-info>/<timestamp>/`. |
| `--no-write` | off | Don't save any weights (for test runs). |
| `--max-train-images N` | all | Use only the first N training images (quick tests; loads them into memory). |
| `--max-val-images N` | all | Same for validation. |
| `--notes "..."` | empty | Free text saved with the run, to remind you what it was for. |

A quick check that everything works (about 2 minutes):

```bash
python -m torch_dnn.train --epochs 3 --max-train-images 1600 --max-val-images 160 --no-write
```

## What a run writes

`data/` is created in the folder you start training from (it is ignored by git):

```
data/model/<timestamp>/saved_model_best_dict.pth   # weights from the epoch with the lowest validation loss
data/model/<timestamp>/last_model_dict.pth         # weights at the end of training
data/logs/<timestamp>/history.jsonl                # one line per epoch (see below)
data/logs/<timestamp>/run_config.json              # every setting, the git commit, the PyTorch version
data/logs/<timestamp>/uncommitted.diff             # code changes not yet committed, if any
data/logs/<timestamp>/events.out.tfevents.*        # TensorBoard logs
```

Use `saved_model_best_dict.pth`. When training finishes it prints the exact commands
to evaluate it and to install it as the default model:

```bash
cp data/model/<timestamp>/saved_model_best_dict.pth ~/.deepcell/models/torch-dnn_<date>.pth
```

`DNN()` (and `torch_dnn.eval` without `--model-path`) loads the `torch-dnn*.pth` in
`~/.deepcell/models` whose name sorts last, so a date in the name keeps the newest one
in front.

## Watching a run

Each epoch prints the training and validation loss, split by head:

```
Training loss: 0.0042  inner_distance=0.0017  outer_distance=0.0019  fgbg=0.0007
Validation loss: 0.0025  inner_distance=0.0010  outer_distance=0.0010  fgbg=0.0006
Epoch time: 102 s (train 81 s)
```

The same numbers, plus the learning rate and the peak GPU memory, are in
`history.jsonl`. For plots, and a picture of the predicted maps for one validation
image every epoch, run `tensorboard --logdir data/logs`.

What to expect:

- The loss falls fast in the first epoch (from about 1 to below 0.05) and then slowly.
  In a test run on the full data the validation loss was 0.0041 after epoch 1 and
  0.0025 after epoch 2.
- **The validation loss is often *lower* than the training loss.** Training images are
  randomly rotated, zoomed and flipped, which makes them harder; validation images are
  not.
- **In a very short run (a few dozen steps) the validation loss can be 10–100 times the
  training loss**, almost all of it from the inner-distance head. That is not a bug.
  Validation runs the network in "evaluation mode", where each batch-normalisation layer
  uses averages collected during training instead of the current batch. In the heads
  those averages update slowly (as in deepcell's Keras model), so for roughly the first
  100 steps they still describe the untrained network. By the end of the first full
  epoch (309 steps) this has gone away.

## The recipe

Everything below matches the deepcell notebook unless it says otherwise.

**Data.** Every image is used at its stored size (512 x 512), including the PC-3 images
(0.55 µm/pixel) and the RPE images (unknown pixel size); deepcell does not rescale them
either. Images without any labelled nucleus are left out. Each image is normalised with
CLAHE (deepcell's `histogram_normalization`), then:

- *training*: a random region is cut out, zoomed by a random factor between 0.75 and
  1.33 (separately for height and width), rotated by a random angle (up to 180°),
  flipped at random, and resized to 256 x 256;
- *validation*: the centre 256 x 256 is cut out. (deepcell takes a random crop instead;
  a fixed crop makes the validation loss comparable between epochs.)

**Targets.** From each label mask: the inner distance (`alpha='auto'`, `beta=1`), the
outer distance (`erosion_width=1`) and a two-class foreground/background map.

**Loss.** Mean squared error on each distance head, plus 0.01 x a class-weighted
cross-entropy on the foreground/background head (the class weights are recomputed for
each batch), summed.

**Optimiser.** Adam, learning rate 1e-4, multiplied by 0.99 after every epoch, gradient
clipping at 0.001 per weight tensor (Keras `clipnorm`). Batch size 16.

**Callbacks.**

- The weights are saved whenever the validation loss reaches a new low.
- Early stopping: training stops if the validation loss has not improved for 10 epochs
  (it cannot trigger in a 16-epoch run that keeps improving).
- "Reduce learning rate on plateau": the learning rate is cut 10x when the validation
  loss has not improved for 5 epochs, and the cut **stays in effect** (the 0.99 decay
  then continues from the lower rate). Each cut is printed and recorded in
  `history.jsonl` (`plateau_lr`). This is a deliberate change from deepcell: in
  TensorFlow 2.8 the per-epoch schedule resets the learning rate at the start of every
  epoch, so the notebook's plateau step never actually took effect.

**Network.** `PanopticNet(**DNN_MODEL_KWARGS)` in `torch_dnn/model.py`: EfficientNetV2-L
backbone starting from ImageNet weights, backbone levels C1–C5, pyramid levels P1–P7,
heads `[1, 1, 2]`. The recipe's constants are at the top of `torch_dnn/train.py` and
`torch_dnn/loaders.py`; to change the recipe, edit them there (they are recorded in
`run_config.json` for every run).

## Old weights

Weights saved by versions of torch_dnn from before November 2026 (for example
`~/.deepcell/dnn/saved_model_best_dict.pth`) belong to a slightly different network and
cannot be loaded. The app and `torch_dnn.eval` stop with an error that says so. A new
model has to be trained.
