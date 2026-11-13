# Tutorial: segmenting nuclei

This walks through segmenting your own nuclear images with `torch_dnn.dnn.DNN`, the
PyTorch version of deepcell's `NuclearSegmentation` application. You need trained
weights first (see [training.md](training.md)); there are no downloadable weights for the
current network.

## 1. Load the model

```python
import numpy as np
import torch
from torch_dnn.dnn import DNN

torch.set_num_threads(4)     # be polite on a shared machine

app = DNN()                  # newest ~/.deepcell/models/torch-dnn*.pth
# or name the weights and the device yourself:
app = DNN(model_path='data/model/20261120093000/saved_model_best_dict.pth', device='cuda')
```

`device` can be `'cpu'`, `'cuda'`, `'cuda:1'` or `'mps'` (Apple). Without it, a GPU is
used if there is one. On a shared server, set `CUDA_VISIBLE_DEVICES` before starting
Python so that `'cuda'` means the GPU you were given.

If no weights are found, or the file is from an older, incompatible version of
torch_dnn, `DNN()` stops with an error that explains what to do.

## 2. Get the image into the right shape

`predict` takes a 4-dimensional array: **(frames, channels, height, width)**, with
exactly **one** channel, the nuclear stain. The "frames" axis can hold the frames of a
time-lapse movie or just a pile of unrelated images; each is segmented on its own.

```python
from tifffile import imread

frame = imread('example/example_image.tiff')   # (512, 512): one image
image = frame[np.newaxis, np.newaxis]           # (1, 1, 512, 512)

movie = imread('my_movie.tif')                  # e.g. (100, 1024, 1024): frames, height, width
image = movie[:, np.newaxis]                    # (100, 1, 1024, 1024)

stack = imread('my_stack.tif')                  # e.g. (100, 1024, 1024, 1): channel last
image = np.moveaxis(stack, -1, 1)               # (100, 1, 1024, 1024)
```

If the shape is wrong, `predict` raises an error that suggests the fix. Raw intensities
are fine (any number type); the app normalises each image itself.

## 3. Segment

```python
masks = app.predict(image, image_mpp=0.65)
masks.shape   # same as image: (frames, 1, height, width)
```

**`image_mpp` is your pixel size in microns per pixel** (check the microscope metadata;
it depends on the objective and camera). The model was trained at 0.65 µm/pixel, so
images are scaled to that size, segmented, and the masks scaled back to your image's
size. Getting it wrong by a lot makes nuclei look too big or too small to the network,
and segmentation suffers. Pass `image_mpp=None` only if your images are already at 0.65.

The result is a *label mask*: 0 is background and each nucleus is filled with its own
whole number. Numbers are only unique within a frame; linking nuclei across frames
(tracking) is a separate step.

```python
n_nuclei = len(np.unique(masks[0, 0])) - 1     # number of nuclei in the first frame
```

Other arguments:

- `batch_size=4`: how many 256 x 256 tiles go through the network at once. Lower it if
  the GPU runs out of memory; raise it for speed.
- `verbose=False`: hide the progress bars.
- `postprocess_kwargs={...}`: change how the network's maps become masks (see below).
- `preprocess_kwargs={'kernel_size': ...}`: the CLAHE normalisation window (default: 1/8
  of the image size, as in deepcell).

## 4. Look at the result

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(frame, cmap='gray')
ax[0].set_title('Nuclear image')
ax[1].imshow(masks[0, 0], cmap='nipy_spectral', interpolation='nearest')
ax[1].set_title('Predicted nuclei')
for a in ax:
    a.set_axis_off()
fig.tight_layout()
plt.show()
```

To draw outlines on the image instead:

```python
from skimage.segmentation import find_boundaries

plt.imshow(frame, cmap='gray')
plt.contour(find_boundaries(masks[0, 0]), levels=[0.5], colors='yellow', linewidths=0.5)
plt.axis('off'); plt.show()
```

## 5. The network's raw output (optional)

```python
masks, maps = app.predict(image, image_mpp=0.65, return_transforms=True)
maps.shape   # (frames, 4, height', width'), at 0.65 µm/pixel (not scaled back)
```

The four channels are the inner distance (0; bright at nucleus centres), the outer
distance (1; above zero inside nuclei), and the background and foreground probabilities
(2 and 3). Looking at channels 0 and 1 helps when masks look wrong: missing seeds in
channel 0 mean missed or merged nuclei; two bright spots in one nucleus mean a split.

## 6. Changing the post-processing

The masks are made from channels 0 and 1 by a seeded watershed (deepcell's
`deep_watershed`). Its settings can be changed per call:

```python
masks = app.predict(image, image_mpp=0.65,
                    postprocess_kwargs={'maxima_threshold': 0.2,    # fewer seeds: fewer splits
                                        'interior_threshold': 0.05,  # tighter masks
                                        'small_objects_threshold': 30})  # drop masks under 30 px
```

The defaults are deepcell's, chosen for its TensorFlow model. After training a new model,
find the best values for it systematically with `torch_dnn.postprocess_sweep`; see
[evaluation.md](evaluation.md), which also explains each setting.

## Large images and memory

Images of any size work: they are cut into overlapping 256 x 256 tiles, and the tiles'
outputs are blended back together before post-processing. All frames are held in memory
at once, so for a long movie of large frames, segment it in chunks:

```python
masks = np.concatenate([app.predict(image[i:i + 50], image_mpp=0.65, verbose=False)
                        for i in range(0, len(image), 50)])
```
