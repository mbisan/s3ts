# /usr/bin/env python3
# # -*- coding: utf-8 -*-

""" File containing the function to download a dataset from UCR/UEA archive. """

# standard library
from pathlib import Path
import logging as log

# basic dependencies
import numpy as np 

# sktime
from sktime.datasets import load_UCR_UEA_dataset


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def download_dataset(dataset: str, dir_cache: Path) -> tuple[np.ndarray, np.ndarray]:

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
    if not dir_cache.exists():
        dir_cache.mkdir(parents=True)

    # Define cache file
    cache_file = dir_cache / f"{dataset}.npz"

    # Check if dataset is already downloaded
    if cache_file.exists():
        # Load dataset from cache
        log.info(f"Loading '{dataset}' from cache...")
        with np.load(cache_file) as data:
            X, Y = data["X"], data["Y"]
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
        Yn = np.zeros_like(Y, dtype=int)
        for i, y in enumerate(np.unique(Y)):
            Yn[Y == y] = i
        Y = Yn

        # Cache dataset
        np.savez_compressed(cache_file, X=X, Y=Y)

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
    return X, Y