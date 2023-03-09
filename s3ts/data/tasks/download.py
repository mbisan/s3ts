from sktime.datasets import load_UCR_UEA_dataset
import numpy as np 

from pathlib import Path
import logging as log

def download_dataset(dataset_name: str, dir_cache: Path) -> None:

    """ Load dataset from UCR/UEA time series archive. """

    log.info(f"Dataset name: {dataset_name}")
    log.info("Creating folders...")
    for path in [dir_cache]:
        path.mkdir(parents=True, exist_ok=True)
        log.info("..." + str(path))
    log.info("Done!")

    cache_file = dir_cache / f"{dataset_name}.npz"
    if cache_file.exists():

        with np.load(cache_file, allow_pickle=True) as data:
            X, Y = data["X"], data["Y"]
            mapping = data["mapping"]

        nsamples = X.shape[0]
        s_length = X.shape[1]
        nclasses = len(np.unique(Y))

        log.info(f"Loading dataset from {str(cache_file)}...")
        log.info(f"Number of samples: {nsamples}")
        log.info(f"Sample length: {s_length}")
        log.info(f"Number of classes: {nclasses}")

    else:

        log.info("Downloading from UCR/UEA...")

        # Download TS dataset from UCR UEA
        X, Y = load_UCR_UEA_dataset(name=dataset_name, 
                                return_type="np2d",
                                return_X_y=True)

        nsamples = X.shape[0]
        s_length = X.shape[1]
        nclasses = len(np.unique(Y))

        log.info(f"Number of samples: {nsamples}")
        log.info(f"Sample length: {s_length}")
        log.info(f"Number of classes: {nclasses}")

        try:
            Y = Y.astype(int)
            Y = Y - Y.min()
            mapping = None
        except ValueError:
            mapping = {k: v for v, k in enumerate(np.unique(Y))}

        np.savez_compressed(cache_file,
            X=X, Y=Y, mapping=mapping)


    return X, Y, mapping