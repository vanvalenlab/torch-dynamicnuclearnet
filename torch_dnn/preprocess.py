"""Convert the DynamicNuclearNet segmentation data (v1.0) from ``.npz`` to Zarr.

The released data are ``train.npz``, ``val.npz`` and ``test.npz`` inside
``DynamicNuclearNet-segmentation-v1_0.zip``, each with

* ``X``: ``(N, 512, 512, 1)`` uint16 nuclear images,
* ``y``: ``(N, 512, 512, 1)`` int32 label images,
* ``meta``: ``(N + 1, 6)`` object array whose first row is the header
  ``[filename, experiment, pixel_size, screening_passed, time_step, specimen]``.

Each split becomes ``<out_dir>/<split>.zarr`` with

* ``X``: ``(N, 1, 512, 512)`` float32 (channels first, the exact uint16 values),
* ``y``: ``(N, 1, 512, 512)`` int32,
* ``meta``: ``(N,)`` structured array with all six metadata columns
  (``pixel_size`` is NaN where the source has none; a missing ``time_step`` is ``''``),

one image per chunk, so the training loader reads one image at a time. No image
is dropped and nothing is resized: deepcell's ``DynamicNuclearNetSegmentation``
loads the splits as they are (``deepcell/datasets/dataset.py``), and the TF
training notebook uses every image at its native resolution. (Most images are
0.65 um/px, but the PC-3 images are 0.55 and the RPE images have no pixel size.)

The arrays inside each ``.npz`` are read a few images at a time, so a split is
never held in memory whole (``train.npz`` holds 7.8 GB of uncompressed arrays).
Each split is written to ``<split>.zarr.partial`` and renamed when complete.

Run::

    nice -n 10 python -m torch_dnn.preprocess            # ~/.deepcell/dnn/*.npz -> ~/.deepcell/dynamic_nuclear_net/
    python -m torch_dnn.preprocess --splits test --overwrite
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import zarr

DATASET_NAME = 'DynamicNuclearNet-segmentation-v1_0'
DEFAULT_SOURCE_DIR = Path.home() / '.deepcell' / 'dnn'
DEFAULT_DATA_DIR = Path.home() / '.deepcell' / 'dynamic_nuclear_net'
SPLITS = ('train', 'val', 'test')

META_COLUMNS = ('filename', 'experiment', 'pixel_size', 'screening_passed', 'time_step', 'specimen')
META_DTYPE = np.dtype([
    ('filename', 'U160'),
    ('experiment', 'U128'),
    ('pixel_size', 'f8'),
    ('screening_passed', 'U32'),
    ('time_step', 'U32'),
    ('specimen', 'U32'),
])

X_DTYPE = np.float32
Y_DTYPE = np.int32


def _create_array(group, name, **kwargs):
    """``group.create_array`` (zarr 3) or ``group.create_dataset`` (zarr 2)."""
    create = getattr(group, 'create_array', None) or group.create_dataset
    return create(name, **kwargs)


def _open_npy_member(zf, name):
    """Open ``name`` inside an ``.npz`` and read its header.

    Returns:
        ``(file, shape, dtype)``; ``file`` is positioned at the first data byte.
    """
    f = zf.open(name)
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    else:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
    if fortran_order:
        f.close()
        raise ValueError(f'{name} is stored in Fortran order; cannot stream it row by row')
    return f, tuple(shape), np.dtype(dtype)


def _read_exact(f, n_bytes):
    """Read exactly ``n_bytes`` from ``f`` (a zip member may return short reads)."""
    parts, remaining = [], n_bytes
    while remaining:
        part = f.read(remaining)
        if not part:
            raise EOFError(f'expected {n_bytes} bytes, got {n_bytes - remaining}')
        parts.append(part)
        remaining -= len(part)
    return b''.join(parts)


def _iter_blocks(zf, name, block):
    """Yield ``(start, rows)`` blocks of an array inside an ``.npz``, ``block`` rows at a time."""
    f, shape, dtype = _open_npy_member(zf, name)
    try:
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        for start in range(0, shape[0], block):
            n = min(block, shape[0] - start)
            buf = _read_exact(f, n * row_bytes)
            yield start, np.frombuffer(buf, dtype=dtype).reshape((n,) + shape[1:])
    finally:
        f.close()


def npz_shape(zf, name):
    """Shape and dtype of an array inside an open ``.npz`` without reading its data."""
    f, shape, dtype = _open_npy_member(zf, name)
    f.close()
    return shape, dtype


def meta_to_structured(meta):
    """Turn the ``.npz`` ``meta`` array (header row first) into a structured array.

    Raises:
        ValueError: the header is not ``META_COLUMNS``.
    """
    meta = np.asarray(meta, dtype=object)
    header = tuple(str(c) for c in meta[0])
    if header != META_COLUMNS:
        raise ValueError(f'unexpected meta header {header}; expected {META_COLUMNS}')
    rows = meta[1:]

    out = np.zeros(len(rows), dtype=META_DTYPE)
    for j, column in enumerate(META_COLUMNS):
        values = rows[:, j]
        if column == 'pixel_size':
            out[column] = [np.nan if v is None else float(v) for v in values]
        else:
            strings = ['' if v is None else str(v) for v in values]
            width = META_DTYPE[column].itemsize // 4
            longest = max((len(s) for s in strings), default=0)
            if longest > width:  # numpy would truncate silently
                raise ValueError(f'meta column {column!r} has a {longest}-character value; the field holds {width}')
            out[column] = strings
    return out


def convert_to_zarr(filename, out_dir=None, overwrite=False, block=32):
    """Convert one split's ``.npz`` to ``<out_dir>/<split>.zarr``.

    Args:
        filename: path to ``train.npz``, ``val.npz`` or ``test.npz``.
        out_dir: output folder (default ``DEFAULT_DATA_DIR``).
        overwrite: replace an existing ``<split>.zarr``; otherwise it is left alone.
        block: images read and written at a time.

    Returns:
        Path of the zarr store.
    """
    filename = Path(filename)
    split = filename.stem
    out_dir = Path(DEFAULT_DATA_DIR if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f'{split}.zarr'
    partial = out_dir / f'{split}.zarr.partial'

    if final.exists() and not overwrite:
        print(f'    {final} exists; skipping (use --overwrite to replace it).')
        return final
    if partial.exists():
        shutil.rmtree(partial)

    with zipfile.ZipFile(filename) as zf:
        x_shape, _ = npz_shape(zf, 'X.npy')
        y_shape, _ = npz_shape(zf, 'y.npy')
        if x_shape[:3] != y_shape[:3] or x_shape[-1] != 1 or y_shape[-1] != 1:
            raise ValueError(f'unexpected shapes X {x_shape}, y {y_shape}')
        n, h, w, _ = x_shape

        with zf.open('meta.npy') as f:
            meta = meta_to_structured(np.lib.format.read_array(f, allow_pickle=True))
        if len(meta) != n:
            raise ValueError(f'{len(meta)} metadata rows for {n} images')

        crcs = {i.filename: format(i.CRC, '08x') for i in zf.infolist()}

        store = zarr.open_group(str(partial), mode='w')
        store.attrs.update({
            'dataset': DATASET_NAME,
            'split': split,
            'source': str(filename),
            'source_member_crc32': crcs,
            'meta_columns': list(META_COLUMNS),
            'layout': 'X (N, 1, H, W) float32; y (N, 1, H, W) int32; channels first',
        })

        print(f'    Writing {split}: {n} images of {h} x {w}.', flush=True)
        for name, dtype in (('X', X_DTYPE), ('y', Y_DTYPE)):
            arr = _create_array(store, name, shape=(n, 1, h, w), chunks=(1, 1, h, w), dtype=dtype)
            for start, rows in _iter_blocks(zf, f'{name}.npy', block):
                # (n, H, W, 1) -> (n, 1, H, W)
                arr[start:start + len(rows)] = np.moveaxis(rows, -1, 1).astype(dtype)

        _create_array(store, 'meta', shape=meta.shape, dtype=meta.dtype)
        store['meta'][:] = meta

    if final.exists():  # only reached with overwrite=True; the new store is complete by now
        shutil.rmtree(final)
    os.replace(partial, final)
    return final


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Convert DynamicNuclearNet segmentation .npz files to Zarr.')
    p.add_argument('--source-dir', default=str(DEFAULT_SOURCE_DIR),
                   help='folder with train.npz / val.npz / test.npz (extracted from the release zip)')
    p.add_argument('--out-dir', default=str(DEFAULT_DATA_DIR), help='where <split>.zarr is written')
    p.add_argument('--splits', nargs='+', default=list(SPLITS), choices=SPLITS)
    p.add_argument('--overwrite', action='store_true', help='replace existing <split>.zarr stores')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for split in args.splits:
        filename = Path(args.source_dir) / f'{split}.npz'
        if not filename.is_file():
            raise FileNotFoundError(
                f'{filename} not found. Extract {DATASET_NAME}.zip (from deepcell) into {args.source_dir}.')
        print(f'Converting {filename}', flush=True)
        print(f'    -> {convert_to_zarr(filename, args.out_dir, overwrite=args.overwrite)}', flush=True)


if __name__ == '__main__':
    main()
