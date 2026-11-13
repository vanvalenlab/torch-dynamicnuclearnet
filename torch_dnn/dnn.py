import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from torch_dnn.model import PanopticNet, DNN_MODEL_KWARGS, DNN_OUTPUT_CHANNELS

from torch_dnn.postprocess_utils import (
    resize_input, resize_output, dnn_preprocess, dnn_postprocess, untile_output, tile_input,
)

# Resolution the network works at, in microns per pixel (deepcell's NuclearSegmentation
# 1.1 config). Images at other resolutions are resized to it and the masks resized back.
MODEL_MPP = 0.65

# deepcell's NuclearSegmentation 1.1 post-processing config
# (deepcell/applications/nuclear_segmentation.py), tuned for deepcell's TF weights and
# kept for reference. That config also passes ``min_distance=10``, a deprecated alias of
# ``radius`` with the same value, so it is left out (it would only raise a
# DeprecationWarning). Every other ``deep_watershed`` setting keeps its default, as in
# deepcell: maxima_smooth=0, interior_smooth=1, fill_holes_threshold=0,
# pixel_expansion=None, label_erosion=0, maxima_algorithm='h_maxima' (so
# ``exclude_border`` has no effect). Head indices are not listed here:
# ``dnn_postprocess`` takes them from ``postprocess_utils.DNN_HEAD_INDICES``.
TF_POSTPROCESS_KWARGS = {
    'radius': 10,
    'maxima_threshold': 0.1,
    'interior_threshold': 0.01,
    'exclude_border': False,
    'small_objects_threshold': 0,
}

# Post-processing defaults for ``DNN.predict``: re-tuned on the validation split with
# ``torch_dnn/postprocess_sweep.py`` for the torch weights trained 2026-11-13 (batch 32,
# 16 epochs). Pooled F1 on validation: 0.9546 with ``TF_POSTPROCESS_KWARGS`` -> 0.9610,
# better for every cell line. Settings within ~0.001 F1 of the best formed a broad
# plateau (radius 2-4, maxima_threshold 0.15-0.25, interior_threshold 0.02-0.03); this
# is its middle. Re-tune after training new weights.
DEFAULT_POSTPROCESS_KWARGS = {
    **TF_POSTPROCESS_KWARGS,
    'radius': 3,
    'maxima_threshold': 0.2,
    'interior_threshold': 0.03,
}

# Where ``DNN()`` looks for weights when no ``model_path`` is given: the newest
# (by name) file matching ``torch-dnn*.pth`` in ``~/.deepcell/models``, e.g.
# ``torch-dnn_2026-11-20.pth``.

_RELEASED_WEIGHTS_REPO = 'vanvalenlab/torch-dynamicnuclearnet'
_RELEASED_WEIGHTS_FILE = 'torch-dnn_2026-11-13.pth'


def _released_weights_path():
    """Download the released weights from Hugging Face if they are not in
    ``~/.deepcell/models`` yet, and return their local path (str)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=_RELEASED_WEIGHTS_REPO,
        filename=_RELEASED_WEIGHTS_FILE,
        local_dir=Path.home() / '.deepcell' / 'models',
    )


def _default_device():
    """``'cuda'`` or ``'mps'`` if available, otherwise ``'cpu'``."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class DNN():
    """The DynamicNuclearNet nuclear segmentation pipeline, after deepcell's
    ``NuclearSegmentation`` application.

    The input is a stack of single-channel nuclear images (for example the
    frames of a time-lapse movie), channels first: ``(frames, 1, x, y)``.

    `predict` runs the whole pipeline: rescale to the model's 0.65 µm/pixel,
    normalise, tile, run the network, stitch the tiles together, turn the
    network output into label masks and resize them back. See the `predict`
    docstring for details.
    """

    def __init__(self, model_path=None, device=None):
        """
        Instantiate the DynamicNuclearNet segmentation pipeline.

        Parameters
        ----------
        model_path : str or pathlib.Path, default=None
            The trained weights: a ``.pth`` file holding a ``PanopticNet``
            state dict, as saved by ``torch_dnn.train``
            (``data/model/<timestamp>/saved_model_best_dict.pth``). If not
            given, the newest ``~/.deepcell/models/torch-dnn*.pth`` is used
            (see `default_model_path`); there are no released weights to
            download.

        device : str, default=None
            A `torch.device` compatible specifier indicating the hardware to be
            used for inference, e.g. ``"cuda"``, ``"cuda:4"``, ``"mps"``, or
            ``"cpu"``. If not specified (the default), a cuda or mps backend is
            used if one is detected, falling back to ``"cpu"`` if not.

        Raises
        ------
        FileNotFoundError
            No ``model_path`` was given and there are no weights in ``~/.deepcell/models``.
        ValueError
            The checkpoint does not fit the network, for example weights of the
            original torch_dnn network (no ``arch_version`` entry) or a Mesmer
            checkpoint.
        """
        self.device = _default_device() if device is None else device

        if model_path is None:
            model_path = _released_weights_path()

        print("Initializing model...")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self._build_network(checkpoint)

        self.model_image_shape = (256, 256)
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.required_rank = len(self.model_image_shape) + 2
        self.required_channels = DNN_MODEL_KWARGS['in_channels']

        self.model_mpp = MODEL_MPP

    def _build_network(self, checkpoint):
        """The ``PanopticNet`` for ``checkpoint``, with its weights loaded strictly,
        on ``self.device`` and in eval mode. Raises ``ValueError`` for a checkpoint
        that does not fit (see ``PanopticNet.load_state_dict``)."""
        if not isinstance(checkpoint, dict):
            raise ValueError(f'Expected a PanopticNet state dict, got a {type(checkpoint).__name__}')
        # use_imagenet=False: every weight is overwritten by load_state_dict
        # below, so downloading the ImageNet backbone would be wasted.
        model = PanopticNet(**DNN_MODEL_KWARGS, use_imagenet=False)
        model.load_state_dict(checkpoint)
        return model.to(self.device).eval()

    def predict(self,
                image,
                image_mpp,
                batch_size=4,
                pad_mode='reflect',
                return_transforms=False,
                preprocess_kwargs=None,
                postprocess_kwargs=None,
                verbose=True):
        """Compute nuclear label masks for `image`.

        The steps: rescale the images to the model's 0.65 µm/pixel, normalise
        them (`dnn_preprocess`: CLAHE, as deepcell's ``histogram_normalization``),
        cut them into 256x256 tiles, run the network, stitch its output back
        together, turn it into label masks (`dnn_postprocess`: ``deep_watershed``
        seeded by the inner distance, masked by the outer distance) and resize
        the masks back to the input size.

        Input images are required to be **channels first**, with 4 dimensions
        ``[batch, channel, x, y]`` and exactly 1 channel. The batch axis can hold
        the frames of a movie; each frame is segmented on its own. Add missing
        axes with ``image[np.newaxis, np.newaxis]`` (one 2D image) or
        ``image[:, np.newaxis]`` (a ``(frames, x, y)`` stack).

        Parameters
        ----------
        image : array_like with shape ``[batch, 1, x, y]``
            The nuclear images to segment.

        image_mpp : float or None
            The scale of the image in microns per pixel. It is required: pass
            ``None`` only to state that the image is already at the model's
            0.65 µm/pixel (no rescaling). The same value applies to every image
            in the batch.

        batch_size : int, default=4
            Number of 256x256 tiles to run through the network at once. This
            controls the memory footprint of inference.

        pad_mode : str, default="reflect"
            Type of padding used when tiling images larger than 256x256 (see
            `numpy.pad`). ``"reflect"`` is deepcell's NuclearSegmentation default.

        return_transforms : bool, default=False
            Whether to return the raw network output (i.e. prior to postprocessing)
            along with the label image.

        preprocess_kwargs : dict, default=None
            Keyword arguments for `dnn_preprocess`: ``kernel_size`` (CLAHE kernel;
            default ``None``, 1/8 of the image size).

        postprocess_kwargs : dict, default=None
            Keyword arguments for ``deep_watershed`` (see `dnn_postprocess`). Keys
            given here override ``DEFAULT_POSTPROCESS_KWARGS``; ``None`` uses those
            defaults.

        verbose : bool, default=True
            Show progress bars for running the network over the tiles and for
            post-processing the frames.

        Returns
        -------
        label_image : numpy.array
            Integer nuclear label masks, shape ``(batch, 1, x, y)``, the size of
            `image`; 0 is background.

        OR

        label_image, output_image : numpy.array
            If ``return_transforms=True``, the raw network output is returned too,
            ``(batch, 4, x', y')`` at the model's resolution (not resized back):
            channel 0 inner distance, 1 outer distance, 2-3 the foreground/background
            probabilities (background, foreground).

        Raises
        ------
        ValueError
            If `image` is not ``(batch, 1, x, y)``, or `image_mpp` is not a positive
            number or None.

        Examples
        --------
        >>> frames = np.random.random((3, 1, 512, 512))
        >>> masks = app.predict(frames, image_mpp=0.65)
        >>> masks.shape
        (3, 1, 512, 512)
        """
        image = np.asarray(image)
        self._check_input(image)
        image_mpp = self._check_mpp(image_mpp)

        # overwrite defaults with any user-provided values
        postprocess_kwargs = {**DEFAULT_POSTPROCESS_KWARGS, **(postprocess_kwargs or {})}

        # Keep track of original shape for rescaling after processing
        orig_img_shape = image.shape

        resized_image = resize_input(image, image_mpp, self.model_mpp)
        image = dnn_preprocess(resized_image, **(preprocess_kwargs or {}))
        tiles, tiles_info = tile_input(image, pad_mode=pad_mode, model_image_shape=self.model_image_shape)
        output_tiles = self._run_network(tiles, batch_size, verbose=verbose)
        output_images = untile_output(output_tiles, tiles_info)

        label_image = dnn_postprocess(output_images, progbar=verbose, **postprocess_kwargs)

        label_image = resize_output(label_image, orig_img_shape).astype(int)

        if return_transforms:
            return label_image, output_images
        else:
            return label_image

    def _check_input(self, image):
        """Raise a ``ValueError``, with a hint on how to fix it, unless ``image`` has
        shape ``(batch, 1, x, y)``."""
        if image.ndim != self.required_rank:
            hint = ''
            if image.ndim == 2:
                hint = ' For one 2D image, use image[np.newaxis, np.newaxis].'
            elif image.ndim == 3:
                hint = ' For a (frames, x, y) stack, use image[:, np.newaxis].'
            raise ValueError(
                f'Input data must have {self.required_rank} dimensions '
                f'[batch, channel, x, y]; got shape {image.shape}.{hint}')
        if image.shape[1] != self.required_channels:
            hint = ''
            if image.shape[-1] == self.required_channels:
                hint = (' This looks like channels-last input; convert it with '
                        'np.moveaxis(image, -1, 1).')
            raise ValueError(
                f'Input data must have {self.required_channels} channel '
                f'(nuclear) in dimension 1 (channels first); '
                f'got shape {image.shape}.{hint}')

    def _check_mpp(self, image_mpp):
        """``image_mpp`` as a float (or None), or a ``ValueError`` if it is not a
        positive number. A missing value (NaN) is rejected rather than guessed."""
        if image_mpp is None:
            return None
        try:
            value = float(image_mpp)
        except (TypeError, ValueError):
            raise ValueError(f'image_mpp must be a number (microns per pixel) or None, '
                             f'got {image_mpp!r}') from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f'image_mpp must be a positive number of microns per pixel, got {image_mpp!r}. '
                f'Pass None if the image is already at the model resolution ({self.model_mpp}).')
        return value

    def _run_network(self, tiles, batch_size, verbose=False):
        """Run the network on ``tiles``, ``(n_tiles, 1, 256, 256)``, ``batch_size`` at a time.

        Returns:
            numpy.array: the network output, ``(n_tiles, 4, 256, 256)``, in the dtype of ``tiles``.
        """
        output_tiles = np.zeros(
            (tiles.shape[0], DNN_OUTPUT_CHANNELS) + self.model_image_shape,
            dtype=tiles.dtype,
        )

        starts = range(0, tiles.shape[0], batch_size)
        for tile_batch_start in tqdm(starts, desc='Running network on tiles', leave=False,
                                     disable=not verbose):
            # Load only this batch to GPU
            tile_batch = torch.tensor(tiles[tile_batch_start:tile_batch_start+batch_size]).to(self.device)

            with torch.inference_mode():
                pred = self.model(tile_batch)

            # Move predictions back to CPU to save GPU memory
            output_tiles[tile_batch_start:tile_batch_start+batch_size] = pred.cpu().numpy()

        return output_tiles
