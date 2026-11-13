# Evaluation and post-processing

Two tools:

- `python -m torch_dnn.eval` scores a model on the **test** split.
- `python -m torch_dnn.postprocess_sweep` searches for the best post-processing
  settings on the **validation** split.

Both need the Zarr files made by `python -m torch_dnn.preprocess` (see
[training.md](training.md)). On the shared server, set `CUDA_VISIBLE_DEVICES` to your GPU
first and pass `--device cuda`.

## Scoring a model on the test split

```bash
python -m torch_dnn.eval --model-path data/model/<timestamp>/saved_model_best_dict.pth --device cuda
```

This segments each of the 717 test images with `DNN.predict`, exactly as you would
segment your own images, and compares the masks with the hand-made ones. On one GPU
it takes a few minutes (20 images take about 10 seconds).

| Option | Default | What it does |
|---|---|---|
| `--model-path` | newest `~/.deepcell/models/torch-dnn*.pth` | Weights to evaluate. |
| `--device` | `cpu` | `cuda` to use the GPU (much faster). |
| `--data-path` | `~/.deepcell/dynamic_nuclear_net/test.zarr` | Which split to score. |
| `--limit N` | all | Score only the first N images (a quick check). |
| `--use-mpp / --ignore-mpp` | `--use-mpp` | Rescale each image from its recorded pixel size to 0.65 µm/pixel (PC-3: 0.55 → 0.65; RPE has none and is not rescaled), or never rescale. |
| `--batch-size` | 4 | 256 x 256 tiles per network pass. |
| `--threads` | 4 | CPU threads for PyTorch. |

It prints a table with a column for all images and one per cell line (`specimen`) and
writes two files to the current folder:

- `eval_results_<time>.csv`: one row per image, with its metadata and every count below;
- `eval_summary_<time>.csv`: the printed table.

### What the numbers mean

Each predicted nucleus is matched to a true one when they overlap enough. Then:

- **Recall**: the fraction of true nuclei that were found.
- **Precision**: the fraction of predicted nuclei that are real.
- **F1**: a single score that balances the two (their harmonic mean). This is the
  headline number.
- **Pooled** means the counts are added up over all images first and the fractions
  computed once, as deepcell reports them. The "mean of per-image" rows average each
  image's score instead; images with few nuclei then count as much as crowded ones.
- **Missed / gained detections**: true nuclei with no prediction, and predictions with
  no true nucleus.
- **Splits**: one true nucleus predicted as several. **Merges**: several true nuclei
  predicted as one. **Many-to-many errors**: tangles of both.
- **Jaccard index** and **SEG**: how well the matched outlines overlap (1 = perfect).

For orientation, the weights trained on 2026-11-13 (16 epochs, batch size 32) score these
pooled F1 values on the whole test split with the default (re-tuned) post-processing:

| All | 3T3 | HEK293 | HeLa | PC-3 | RAW264 | RPE |
|---|---|---|---|---|---|---|
| 0.959 | 0.987 | 0.963 | 0.928 | 0.969 | 0.955 | 0.985 |

With deepcell's original settings (`TF_POSTPROCESS_KWARGS`) the same weights score 0.950
overall. Most of HeLa's errors come from one movie, `HeLa-S3_nuc_pos16_q2_5c.trk` (test
images 126–167): in its frames with a brighter, noisier background, the model marks
patches of background as nuclei. Without that movie, HeLa scores 0.960. Look at each cell
line, not only the overall score, since they differ in difficulty.

## Re-tuning the post-processing

The network outputs two maps per image, and a separate step (deepcell's
`deep_watershed`) turns them into masks: peaks of the inner-distance map become seeds,
one per nucleus, and each seed grows to fill the region where the outer-distance map is
high enough. The main settings are:

| Setting | Default | Meaning |
|---|---|---|
| `maxima_threshold` | 0.2 | How prominent a peak must be to become a seed. Higher: fewer seeds (fewer splits, more merges). |
| `radius` | 3 | Radius (pixels, at 0.65 µm/pixel) of the disk used to look for peaks; roughly, peaks closer than this merge into one seed. |
| `interior_threshold` | 0.03 | Outer-distance value above which a pixel can belong to a nucleus. Higher: smaller masks, and dim nuclei can be lost. |
| `maxima_smooth`, `interior_smooth` | 0, 1 | Gaussian smoothing of each map before use. |
| `small_objects_threshold` | 0 | Remove masks smaller than this many pixels. |
| `fill_holes_threshold` | 0 | Fill holes in masks smaller than this. |
| `pixel_expansion` | none | Grow every mask by this many pixels. |

The defaults (`DEFAULT_POSTPROCESS_KWARGS` in `torch_dnn/dnn.py`) were re-tuned this way
for the weights trained on 2026-11-13; deepcell's values for its TensorFlow model (radius
10, `maxima_threshold` 0.1, `interior_threshold` 0.01) are kept as `TF_POSTPROCESS_KWARGS`.
A newly trained model's maps are similar but not identical, so after training new
weights **search for better settings on the validation split** (never on the test split,
or the test score is no longer a fair estimate).

### Step 1: run the network once

```bash
python -m torch_dnn.postprocess_sweep cache \
    --model-path data/model/<timestamp>/saved_model_best_dict.pth \
    --out data/sweep/cache_val --device cuda
```

This saves the two maps for every validation image (`--data-path` to use another split,
`--limit N` for only the first N images). For all 1417 images it takes a few minutes
and about 3 GB of disk.

### Step 2: try the settings

Write a grid file listing the values to try. Every combination within an entry is run;
settings an entry does not mention keep their defaults. For example `grid.json`:

```json
[
  {"label": "thresholds", "grid": {"maxima_threshold": [0.05, 0.1, 0.2],
                                   "interior_threshold": [0.01, 0.05, 0.1]}},
  {"label": "radius", "grid": {"radius": [3, 5, 10]}}
]
```

```bash
python -m torch_dnn.postprocess_sweep sweep --cache data/sweep/cache_val \
    --grid grid.json --out data/sweep/results.csv --workers 4
```

It prints the 10 best settings by pooled F1 and writes every setting's recall,
precision, F1 and error counts to `results.csv` (plus `results.json`, a record of what
was run, and `results.per_image.npz`, the per-image counts). `--subset 300` scores a
random 300 images, for a faster first look. Only the cheap post-processing step is
repeated, so a few dozen settings take minutes, not hours. The masks are exactly what
`DNN.predict` would produce with those settings.

### Step 3: use the winner

For one-off use, pass the settings to the app:

```python
masks = app.predict(image, image_mpp=0.65,
                    postprocess_kwargs={'maxima_threshold': 0.2, 'interior_threshold': 0.1})
```

To make them the defaults (for the app and for `torch_dnn.eval`), edit
`DEFAULT_POSTPROCESS_KWARGS` in `torch_dnn/dnn.py`, then score the test split once with
`torch_dnn.eval` to report the final numbers.
