# /usr/bin/env python3
# # -*- coding: utf-8 -*-

""" File containing the function to download a dataset from UCR/UEA archive. """

# standard library
from s3ts.data.series import compute_medoids
from pathlib import Path
import logging as log
import warnings

# basic dependencies
import numpy as np 

# sktime
warnings.filterwarnings("ignore")
from sktime.datasets import load_UCR_UEA_dataset


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def download_dataset(dataset: str, storage_dir: Path) -> tuple[np.ndarray, np.ndarray]:

    """ Load dataset from UCR/UEA time series archive. 
    If the dataset is not already downloaded, it will be downloaded and cached.
    If the dataset is already downloaded, it will be loaded from the cache.
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to download.
    dir_cache : Path
        Path to the cache directory.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the dataset features and labels.
    """

    # Create cache directory if it does not exist
    datasets_dir = storage_dir / "datasets"
    
    if not datasets_dir.exists():
        datasets_dir.mkdir(parents=True)

    # Define cache file
    dset_file = datasets_dir /f"{dataset}.npz"

    # Check if dataset is already downloaded
    if dset_file.exists():
        # Load dataset from cache
        log.info(f"Loading '{dataset}' from cache...")
        with np.load(dset_file) as data:
            X, Y = data["X"], data["Y"] 
            medoids, medoid_idx = data["medoids"], data["medoid_idx"]
    else:
        # Download TS dataset from UCR UEA
        log.info(f"Downloading '{dataset}' from UCR/UEA...")
        X, Y = load_UCR_UEA_dataset(name=dataset, 
                                return_type="np2d",
                                return_X_y=True)
        X: np.ndarray = X.astype(np.float32)
        Y: np.ndarray = Y.astype(np.int8)

        # Ensure the dataset is normalized
        X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

        # Ensure labels are integers
        Yn = np.zeros_like(Y, dtype=np.int8)
        for i, y in enumerate(np.unique(Y)):
            Yn[Y == y] = i
        Y = Yn

        # exceptions
        X, Y = dset_exceptions(dataset, X, Y)
        medoids, medoid_idx = compute_medoids(X, Y)

        # Cache dataset
        np.savez_compressed(dset_file, X=X, Y=Y, medoids=medoids, medoid_idx=medoid_idx)

    # Get the number of classes
    n_classes = len(np.unique(Y))

    # Get the number of samples
    n_samples = X.shape[0]

    # Get the length of each sample
    s_length = X.shape[1]

    # Do some logging
    log.info(f"Number of samples: {n_samples}")
    log.info(f"Number of classes: {n_classes}")
    log.info(f"Sample length: {s_length}")

    # Return dataset features and labels
    return X, Y, medoids, medoid_idx

def dset_exceptions(dataset: str, X: np.ndarray, Y: np.ndarray):

    """ Exceptions for datasets. """

    if dataset == "Plane":

        # remove class 4
        for i in [4]:
            X = X[Y != i]
            Y = Y[Y != i]

        # ensure label consistency
        Yn = np.zeros_like(Y, dtype=np.int8)
        for i, y in enumerate(np.unique(Y)):
            Yn[Y == y] = i
        Y = Yn

    elif dataset == "Trace":
        
        # remove classes 2 and 3
        X = X[Y != 2]
        Y = Y[Y != 2]
        X = X[Y != 3]
        Y = Y[Y != 3]

        # ensure label consistency
        Yn = np.zeros_like(Y, dtype=np.int8)
        for i, y in enumerate(np.unique(Y)):
            Yn[Y == y] = i
        Y = Yn
    
    elif dataset == "OSULeaf":
        
        # remove class 5
        X = X[Y != 5]
        Y = Y[Y != 5]

        # ensure label consistency
        Yn = np.zeros_like(Y, dtype=np.int8)
        for i, y in enumerate(np.unique(Y)):
            Yn[Y == y] = i
        Y = Yn

    return X, Y