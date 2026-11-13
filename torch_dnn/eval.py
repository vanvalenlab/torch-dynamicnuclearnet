"""Evaluate DynamicNuclearNet weights on the DynamicNuclearNet test split.

Each test image is segmented with ``DNN.predict`` and scored against its ground
truth with ``torch_dnn.metrics``; recall, precision and F1 are pooled over all
images, as deepcell's ``Metrics.summarize_object_metrics_df`` reports them::

    python -m torch_dnn.eval --model-path data/model/<timestamp>/saved_model_best_dict.pth \\
        --device cuda --data-path ~/.deepcell/dynamic_nuclear_net/test.zarr

Images are read from the Zarr file one at a time, so the split is never held in
memory.
"""

from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import tqdm
import click

from skimage.color import label2rgb
from skimage.exposure import rescale_intensity

from torch_dnn.metrics import Metrics

# Keys reported in the summary. recall, precision and f1 are pooled over all
# images (micro-averaged), which is how deepcell's
# ``Metrics.summarize_object_metrics_df`` reports them.
SUMMARY_KEYS = [
    'recall', 'precision', 'f1',
    'n_true', 'n_pred', 'correct_detections',
    'gained_detections', 'missed_detections', 'split', 'merge', 'catastrophe',
    'jaccard', 'seg', 'seg_full',
]

SUMMARY_LABELS = {
    'n_images': 'Images',
    'recall': 'Recall (pooled)',
    'precision': 'Precision (pooled)',
    'f1': 'F1 (pooled)',
    'mean_per_image_recall': 'Recall (mean of per-image)',
    'mean_per_image_precision': 'Precision (mean of per-image)',
    'mean_per_image_f1': 'F1 (mean of per-image)',
    'n_true': 'True objects',
    'n_pred': 'Predicted objects',
    'correct_detections': 'Correct detections',
    'gained_detections': 'Gained detections',
    'missed_detections': 'Missed detections',
    'split': 'Split events',
    'merge': 'Merge events',
    'catastrophe': 'Many-to-many errors',
    'jaccard': 'Jaccard index (mean of per-image)',
    'seg': 'SEG, 1:1 matches only (mean of per-image)',
    'seg_full': 'SEG, standard (pooled over true objects)',
}

# Written by ``python -m torch_dnn.preprocess`` (from ~/.deepcell/dnn/test.npz).
DEFAULT_DATA_PATH = Path.home() / '.deepcell' / 'dynamic_nuclear_net' / 'test.zarr'


def score_images(y_pred, y_true, progbar=False):
    """Per-image object metrics.

    Args:
        y_pred (numpy.array): predicted label images, shape (batch, x, y).
        y_true (numpy.array): ground-truth label images, same shape.
        progbar (bool): show a progress bar.

    Returns:
        pandas.DataFrame: one row of ``ObjectMetrics`` counts and scores per image.
    """
    return Metrics('torch_dnn').calc_object_stats(y_true, y_pred, progbar=progbar)


def summarize(per_image):
    """The summary over a table of per-image metrics.

    ``recall``, ``precision`` and ``f1`` are **pooled**: correct detections,
    true objects and predicted objects are summed over all images before
    dividing, and ``f1`` is the harmonic mean of the pooled recall and
    precision, as in deepcell. The ``mean_per_image_*`` keys give the mean of
    the per-image scores instead, for comparison only. Counts are summed over
    images.

    Args:
        per_image (pandas.DataFrame): rows from `score_images`.

    Returns:
        dict: ``n_images``, the ``SUMMARY_KEYS`` and the ``mean_per_image_*`` scores.
    """
    pooled = Metrics('torch_dnn').summarize_object_metrics_df(per_image)
    summary = {'n_images': len(per_image), **{k: pooled[k] for k in SUMMARY_KEYS}}
    for k in ('recall', 'precision', 'f1'):
        summary[f'mean_per_image_{k}'] = float(per_image[k].mean())
    return summary


def evaluate(y_pred, y_true, progbar=False):
    """Score a batch of label images against the ground truth.

    Args:
        y_pred (numpy.array): predicted label images, shape (batch, x, y).
        y_true (numpy.array): ground-truth label images, same shape.
        progbar (bool): show a progress bar.

    Returns:
        tuple(pandas.DataFrame, dict): one row of object metrics per image, and
        the summary over the batch (see `summarize`).
    """
    per_image = score_images(y_pred, y_true, progbar=progbar)
    return per_image, summarize(per_image)


def summarize_by_group(per_image, group_column='specimen'):
    """Summaries over all images and over each value of ``group_column``.

    Args:
        per_image (pandas.DataFrame): per-image metrics with a ``group_column`` column.
        group_column (str): e.g. ``'specimen'`` (the cell line in the DynamicNuclearNet data).

    Returns:
        pandas.DataFrame: one row per group, ``'all'`` first, indexed by ``group``.
    """
    rows = {'all': summarize(per_image)}
    if group_column in per_image:
        for name, df in per_image.groupby(group_column, sort=True):
            rows[f'{group_column}={name}'] = summarize(df)
    out = pd.DataFrame.from_dict(rows, orient='index')
    out.index.name = 'group'
    return out


def pretty_print_summary(summaries: pd.DataFrame):
    """Print a summary table: one row per metric, one column per group."""
    order = [k for k in SUMMARY_LABELS if k in summaries.columns]
    # Format before transposing so counts print as whole numbers.
    fmt = lambda v: f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v)
    table = summaries[order].map(fmt).T.rename(index=SUMMARY_LABELS)

    print('-'*10 + ' Summary ' + '-'*10)
    print('Pooled = counts summed over all images before dividing (as in deepcell).')
    print()
    print(table.to_string())

    return None


def create_overlays(x, gt, pred):
    x = np.squeeze(x)
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)

    # Rescale raw data
    percentiles = np.percentile(x[np.nonzero(x)], [5, 95])
    raw = rescale_intensity(
        x, in_range=(percentiles[0], percentiles[1]), out_range="float32"
    )

    # Overlay gt on raw
    gt_overlay = label2rgb(gt, image=raw, bg_label=0)

    # Overlay pred on raw
    pred_overlay = label2rgb(pred, image=raw, bg_label=0)

    return gt_overlay, pred_overlay


def image_mpp_or_none(value):
    """A ``pixel_size`` entry of the test metadata as ``DNN.predict``'s ``image_mpp``:
    the value as a float, or ``None`` (no rescaling) when it is missing (NaN)."""
    value = float(value)
    return None if np.isnan(value) else value


def run_evaluation(app, data_path, use_mpp=True, batch_size=4, limit=None):
    """Segment every image of a split with ``app`` and score it.

    Args:
        app: a ``torch_dnn.dnn.DNN`` (or anything with the same ``predict``).
        data_path: Zarr file with ``X`` and ``y`` of shape ``(N, 1, H, W)`` and ``meta``
            (a structured array with ``pixel_size``, ``specimen``, ...), as written by
            ``torch_dnn.preprocess`` to ``~/.deepcell/dynamic_nuclear_net/test.zarr``.
        use_mpp (bool): pass each image's ``pixel_size`` to ``predict`` (missing
            values mean no rescaling). With False, no image is rescaled.
        batch_size (int): tiles per network batch.
        limit (int): score only the first ``limit`` images.

    Returns:
        pandas.DataFrame: per-image metrics, with ``image`` and the metadata columns first.
    """
    import zarr

    z = zarr.open(str(data_path), mode='r')
    # Zarr cannot index a structured array directly: read it into numpy first.
    meta = pd.DataFrame(z['meta'][:]) if 'meta' in z else pd.DataFrame(index=range(z['X'].shape[0]))
    n = z['X'].shape[0] if limit is None else min(limit, z['X'].shape[0])

    rows = []
    for i in tqdm.tqdm(range(n)):
        x = z['X'][i:i + 1]
        y_true = np.asarray(z['y'][i:i + 1, 0]).astype(int)
        mpp = image_mpp_or_none(meta['pixel_size'].iloc[i]) if use_mpp and 'pixel_size' in meta else None
        y_pred = app.predict(x, image_mpp=mpp, batch_size=batch_size, verbose=False)[:, 0]
        with warnings.catch_warnings():
            # relabelling warnings are expected (ground truth is not always sequential)
            warnings.simplefilter('ignore', UserWarning)
            rows.append(score_images(y_pred, y_true))

    per_image = pd.concat(rows, ignore_index=True)
    info = meta.iloc[:n].reset_index(drop=True)
    per_image = pd.concat([info, per_image], axis=1)
    per_image.insert(0, 'image', np.arange(n))
    return per_image


@click.command()
@click.option(
    '--device',
    default='cpu',
    help="""The device that you want to run the inference on.
            Can be `'cpu'` or `'cuda'`. If `'cuda'`, can also specify the specific GPU if multiple are available."""
)
@click.option(
    '--model-path',
    default=None,
    help="""Path to model weights (.pth).
            If unset, will select the latest available torch-dnn*.pth from
            the canonical Deepcell models location ($HOME/.deepcell/models)"""
)
@click.option(
    '--data-path',
    default=str(DEFAULT_DATA_PATH),
    show_default=True,
    help="""Path to the Zarr file containing the test data for evaluation."""
)
@click.option('--batch-size', default=4, show_default=True, help='Tiles per network batch.')
@click.option('--use-mpp/--ignore-mpp', default=True, show_default=True,
              help="Rescale each image from its metadata pixel_size to the model's 0.65 µm/px "
                   "(missing values: no rescaling), or never rescale.")
@click.option('--limit', default=None, type=int, help='Evaluate only the first N images.')
@click.option('--threads', default=4, show_default=True, help='CPU threads for torch (shared server).')
def main(device, model_path, data_path, batch_size, use_mpp, limit, threads):
    """Segment every test image and score the masks against the ground truth.

    Writes ``eval_results_<timestamp>.csv`` (one row per image) and
    ``eval_summary_<timestamp>.csv`` (pooled over all images and per specimen)
    to the current directory, and returns ``(summaries, per_image)``.
    """
    import torch
    from torch_dnn.dnn import DNN, default_model_path

    torch.set_num_threads(threads)
    if model_path is None:
        model_path = default_model_path()
    print(f"Evaluating: {model_path}")

    app = DNN(model_path=model_path, device=device)
    per_image = run_evaluation(app, data_path, use_mpp=use_mpp, batch_size=batch_size, limit=limit)
    summaries = summarize_by_group(per_image, 'specimen')

    pretty_print_summary(summaries)

    stamp = datetime.now().isoformat(timespec="seconds")
    per_image.to_csv(f'eval_results_{stamp}.csv', index=False)
    summaries.to_csv(f'eval_summary_{stamp}.csv')
    return summaries, per_image


if __name__ == "__main__":
    main()
