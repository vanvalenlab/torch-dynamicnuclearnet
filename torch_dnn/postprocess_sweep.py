import zarr
import numpy as np
import itertools
import random

from torch_dnn.dnn import DNN
from torch_dnn.metrics import Metrics

from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from scipy.stats import hmean
from tqdm import tqdm
from pathlib import Path
import pandas as pd


def make_sweep(sweep, default_kwargs):

    sweep_list = []

    sweep_names = [*sweep.keys()]
    sweep_vals = [*sweep.values()]

    for combos in itertools.product(*sweep_vals):
        curr_kwargs = default_kwargs.copy()

        for i, sweep_name in enumerate(sweep_names):
            curr_kwargs[sweep_name] = combos[i]

        sweep_list.append(curr_kwargs)

    return sweep_list
        
    

def evaluate(y_pred, y_test):
    m = Metrics("DVC Mesmer")
    metrics = m.calc_object_stats(y_test, y_pred, progbar=False)

    # calculate image-level recall and precision for F1 score
    recall = metrics["correct_detections"].values / np.where(metrics["n_true"].values==0, 1, metrics["n_true"].values)

    precision = metrics["correct_detections"] / np.where(metrics["n_pred"].values==0, 1, metrics["n_pred"].values)
    f1 = hmean([recall, precision])

    # record summary stats
    summary = m.summarize_object_metrics_df(metrics)

    valid_keys = {
        "recall",
        "precision",
        "jaccard",
        "n_true",
        "n_pred",
        "gained_detections",
        "missed_detections",
        "split",
        "merge",
        "catastrophe",
    }

    output_data = {}
    for k in valid_keys:
        if k in {"jaccard", "recall", "precision"}:
            output_data[k] = float(summary[k])
        else:
            output_data[k] = int(summary[k])
    output_data["f1"] = float(np.mean(f1))

    return output_data

def main():

    output_metrics = {
        "recall": [],
        "precision": [],
        "jaccard": [],
        "n_true": [],
        "n_pred": [],
        "gained_detections": [],
        "missed_detections": [],
        "split": [],
        "merge": [],
        "catastrophe": [],
        "f1": [],
        'interior_threshold': [],
        'interior_smooth': [],
        'maxima_threshold': [],
        'radius': [],
        'maxima_smooth': [],
        'small_objects_threshold': [],
        'fill_holes_threshold': [],
        'pixel_expansion': []
    }

    config = {
        'data_path': Path.home() / '.deepcell/dnn/test.zarr',
        'model_path': 'data/model/20260505104758/saved_model_best_dict.pth'
    }

    # Whole cell, nuc

    sweep_classical = {
        'interior_threshold': [0.05, 0.075],
        'interior_smooth': [0, 1],
        'pixel_expansion': [0, 1],
        'maxima_threshold': [0.075, 0.1],
        'maxima_smooth': [0, 1],
    }

    default_kwargs = {
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'maxima_threshold': 0.1,
            'maxima_smooth': 1,
            'interior_threshold': 0.1,
            'interior_smooth': 0.5,
            'radius': 2
        }

    all_sweeps = make_sweep(sweep_classical, default_kwargs)
        
    z_test = zarr.open(f"{config['data_path']}")
    random_indices = random.sample(range(z_test['X'].shape[0]), 50)

    X_test = z_test['X'][random_indices]
    y_test = z_test['y'][random_indices].astype(int)
    mpps = z_test['meta']['pixel_size'][random_indices]

    # Load model and application
    model = DNN(
        model_path = config['model_path'],
        device='cuda:1',
    )
    
    # evaluate the model
    for curr_sweep in tqdm(all_sweeps):

        pred_temp = np.zeros_like(X_test, dtype=int)

        for i in tqdm(range(X_test.shape[0]),leave=False):

            pred_temp[i] = model.predict(X_test[i:i+1], 
                                  image_mpp=mpps[i], 
                                  postprocess_kwargs=curr_sweep,
                                  return_transforms=False,
                                  batch_size=10)

        curr_metrics = evaluate(pred_temp[:], y_test[:])

        for k, v in curr_metrics.items():
            output_metrics[k].append(v)
        for k, v in curr_sweep.items():
            output_metrics[k].append(v)
            
    processed_metrics = pd.DataFrame(output_metrics)
    processed_metrics.to_csv('parameter_sweep_metrics.csv')

if __name__ == "__main__":
    main()
