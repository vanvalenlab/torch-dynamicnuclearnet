import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zarr
from torch_dnn.transforms import transform_masks
from torch_dnn.augmentation import MultiTransform
from torch_dnn.utils import histogram_normalization, percentile_threshold

class DNNDataset(Dataset):

    def __init__(self, 
                 X, 
                 y,
                 mpps,
                 dataset_type='train',
                 in_transforms = ["inner-distance", "outer-distance", "fgbg"], 
                 augment=True, 
                 crop_size = 256,
                 rotation_range=180,
                 zoom=0.75,
                 preprocess=False,
                 target_mpp = 0.65,
                outer_erosion_width=1,
                inner_distance_alpha="auto",
                inner_distance_beta=1,
                inner_erosion_width=0,
                n_semantic_heads = [1,1,2]
                 ):
    
        self.mpps = mpps
        self.X = X
        self.y = y

        if self.mpps is not None:
            self.mpps = np.where(np.isnan(self.mpps), 0.55, self.mpps)
        
        self.in_transforms = in_transforms
        self.augment = augment
        self.dataset_type = dataset_type
        self.transform_type = [
            'bilinear',
            'bilinear',
            'bilinear',
            'nearest',
            'nearest'
        ]
        self.n_semantic_heads = n_semantic_heads
        self.transforms_kwargs = {

            "outer-distance": {
                "erosion_width": outer_erosion_width
                },

            "inner-distance": {
                "alpha": inner_distance_alpha,
                "beta": inner_distance_beta,
                "erosion_width": inner_erosion_width,
                },

        }

        self.crop_size = crop_size
        self.rotation_range = rotation_range
        self.zoom = zoom
        self.preprocess = preprocess
        self.target_mpp = target_mpp

    def _normalize(self, X):

        X = np.expand_dims(X, 0)

        assert len(X.shape) == 4, 'add batch dimension'
        X_norm = percentile_threshold(X, percentile=99.9)
        X_norm = histogram_normalization(X_norm, kernel_size=128)

        # Return without batch dimension for dataloader
        X_norm = X_norm.squeeze(axis=0)

        return X_norm

    def _transform_labels(self, y):
        
        C, H, W = y.shape

        y_semantic = np.zeros((sum(self.n_semantic_heads), H, W)).astype(y.dtype)

        for i, (transform, head_len) in enumerate(zip(self.in_transforms, self.n_semantic_heads)):
            
            transform_kwargs = self.transforms_kwargs.get(transform, dict())
            y_semantic[i:i+head_len] = transform_masks(y, transform, unbatched=True,
                                            **transform_kwargs)

        return y_semantic

    def __len__(self):
            return self.X.shape[0]

    def __getitem__(self, idx):

        # Indexing for histogram normalization allows for no batches
        x = self.X[idx]
        y = self.y[idx]

        mpp = self.mpps[idx]

        x = self._normalize(x)

        y = self._transform_labels(y)

        # Convert to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        combined = torch.cat([x, y], dim=0)
        combined_out = torch.zeros((combined.shape[0], self.crop_size, self.crop_size))
        
        transform = MultiTransform(mpp=mpp, target_mpp=self.target_mpp, dataset_type=self.dataset_type)

        # Stack x and y along channel dimension
        for c in range(combined.shape[0]):
            combined_out[c] = transform(combined[c], interpolation_mode=self.transform_type[c])
        
        # Split back into x and y
        x_out = combined_out[:x.shape[0]]
        y_out = combined_out[x.shape[0]:]

        # Undo background padding issues -- resets background padding as 1
        y_out[3] = y_out[2] < 1

        return (x_out, y_out)
    
def create_data_loaders(
    train,
    val,
    crop_size=256,
    zoom_min=0.75,
    batch_size=16,
    num_workers=4,
):

    dataloader = None
    valloader = None

    if train is not None:

        train_dataset = DNNDataset(
            train['X'], 
            train['y'],
            train['meta']['pixel_size'],
            crop_size=crop_size,
            dataset_type='train',
            zoom=zoom_min,
            augment=True)
        
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    if val is not None:
        val_dataset = DNNDataset(
            val['X'], 
            val['y'],
            val['meta']['pixel_size'],
            crop_size=crop_size,
            dataset_type='val',
            zoom=zoom_min,
            augment=True)  
    
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return dataloader, valloader

if __name__ == '__main__':

    z_train = zarr.open("/data/shared/tissuenet/tissuenet_v1.1_train.zarr")
    z_val = zarr.open("/data/shared/tissuenet/tissuenet_v1.1_val.zarr")

    # Set up data generators with updated data
    train_data, val_data = create_data_loaders(
        z_train,
        z_val,
        crop_size=256,
        zoom_min=0.75,
        batch_size=1,
        num_workers=1,
        semantic_heads = [1,3,1,3]
    )

    train_iter = iter(train_data)
    sample = next(train_iter)
    print(sample[0].shape, sample[1].shape)


    # train_iter = iter(val_data)
    # sample = next(train_iter)
    # print(sample[0].shape, sample[1].shape)
