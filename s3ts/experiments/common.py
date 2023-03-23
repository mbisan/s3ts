#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Common functions for the experiments. """

# standard library
from pathlib import Path
from math import ceil
import logging as log

# basics
import numpy as np
import pandas as pd

# models / modules
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

# data processing stuff
from scipy.spatial import distance_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sktime.clustering.k_medoids import TimeSeriesKMedoids

# in-package imports
from s3ts.data.modules import FullDataModule
from s3ts.models.wrapper import WrapperModel
from s3ts.data.oesm import compute_DM

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_medoids(
        X: np.ndarray, 
        Y: np.ndarray,
        distance_type: str = 'dtw'
    ) -> tuple[np.ndarray, np.ndarray]: 

    """ Computes the medoids of the classes in the dataset. 
    
    Parameters
    ----------
    X : np.ndarray
        The time series dataset.
    Y : np.ndarray
        The labels of the time series dataset.
    distance_type : str, optional
        The distance type to use, by default 'dtw'
    """

    # Check the distance type
    if distance_type not in ["euclidean", "dtw"]:
        raise ValueError("The distance type must be either 'euclidean' or 'dtw'.")
    
    # Check the shape of the dataset and labels match
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of samples in the dataset and labels must be the same.")

    # Get the number of classes
    n_classes = len(np.unique(Y))
    
    # Get the length of the time series
    s_length = X.shape[1]

    # Initialize the arrays
    medoids = np.empty((n_classes, s_length), dtype=float)
    medoid_ids = np.empty(n_classes, dtype=int)
    
    # Find the medoids for each class
    for i, y in enumerate(np.unique(Y)):

        # Get the samples of the class
        index = np.argwhere(Y == y)
        Xy = X[index, :]

        # ...using Euclidean distance        
        if distance_type == "euclidean":
            medoid_idx = np.argmin(distance_matrix(Xy.squeeze(), Xy.squeeze()).sum(axis=0))
            medoids[i,:] = Xy[medoid_idx,:]
            medoid_ids[i] = index[medoid_idx]

        # ...using Dynamic Time Warping (DTW)
        if distance_type == "dtw":
            if Xy.shape[0] > 1:
                tskm = TimeSeriesKMedoids(n_clusters=1, init_algorithm="forgy", metric="dtw")
                tskm.fit(Xy)
                medoids[i,:] = tskm.cluster_centers_.squeeze()
                medoid_ids[i] = np.where(np.all(Xy.squeeze() == medoids[i,:], axis=1))[0][0]
            else:
                medoids[i,:] = Xy.squeeze()
                medoid_ids[i] = np.where(np.all(Xy.squeeze() == medoids[i,:], axis=1))[0][0]

    # Return the medoids and their indices
    return medoids, medoid_ids

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_STS(
        X: np.ndarray, 
        Y: np.ndarray,
        STS_samples: int,
        shift_limits: bool,
        mode: str = "random",
        random_state: int = 0,
        ) -> tuple[np.ndarray, np.ndarray]:

    """ Generates a Streaming Time Series (STS) from a given dataset. """

    # Check the shape of the dataset and labels match
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of samples in the dataset and labels must be the same.")

    # Set the random state for reproducibility
    rng = np.random.default_rng(seed=random_state)
    
    # Get the number of classes
    n_classes = len(np.unique(Y))
    
    # Get the length of the time series
    s_length = X.shape[1]
    
    # Get the number of samples
    n_samples = X.shape[0]

    # Get the length of the final STS
    STS_length = STS_samples*s_length

    # Do some logging
    log.info(f"Number of samples: {n_samples}")
    log.info(f"Length of samples: {s_length}")
    log.info(f"Number of classes: {n_classes}")
    log.info(f"Class ratios: {np.unique(Y, return_counts=True)[1]/n_samples}")
    log.info(f"Length of STS: {STS_length}")

    # Initialize the arrays
    STS = np.empty(STS_length, dtype=float)
    SCS = np.empty(STS_length, dtype=int)

    # Generate the STS 
    if mode == "random":
        for s in range(STS_samples):
            random_idx = rng.integers(0, n_samples)

            # Calculate shift so that sample ends match
            shift = STS[s-1] - X[random_idx,0] if shift_limits else 0

            STS[s*s_length:(s+1)*s_length] = X[random_idx,:] + shift
            SCS[s*s_length:(s+1)*s_length] = Y[random_idx]

    # Normalize the STS
    STS = (STS - np.mean(STS))/np.std(STS)

    # Return the STS and the SCS
    return STS, SCS


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def prepare_dm(
        dataset: str, 
        X_train: np.ndarray, X_pretest: np.ndarray, 
        Y_train: np.ndarray, Y_pretest: np.ndarray,
        STS_sample_multiplier: int, 
        train_samples_per_class: int,
        rho_dfs: float, batch_size: int, 
        window_length: int, window_stride: int,
        fold_number: int, random_state: int,
        num_workers: int = 4,
        pattern_type: str = "medoids",
        dir_cache: Path = Path("cache"),
        test_sample_multiplier: int = 2,
        pretrain_sample_multiplier: int = 16,
        ) -> tuple[FullDataModule, FullDataModule]:

    """ Prepare the data module for training/pretraining/testing. """


    # Validate the inputs
    n_classes = len(np.unique(Y_train))     # Get the number of classes
    s_length = X_train.shape[1]             # Get the length of the time series
    n_samp_tra = X_train.shape[0]           # Get the number of train samples
    n_samp_pre = X_pretest.shape[0]         # Get the number of pretest samples

    # Check the pattern type is valid
    valid_patterns = ["medoids"]
    if pattern_type not in valid_patterns:
        raise ValueError(f"patterns must be one of {valid_patterns}")

    # Check there is the same numbe of classes in train and test
    if len(np.unique(Y_train)) != len(np.unique(Y_pretest)):
        raise ValueError("The number of classes in train and test must be the same.")
    
    # Check the number of samples per class in train
    if np.unique(Y_train, return_counts=True)[1].min() < train_samples_per_class:
        raise ValueError(f"The number of samples per class in the train set must be at least {train_samples_per_class}.")

    # Check the number of samples per class in pretest
    if np.unique(Y_pretest, return_counts=True)[1].min() < train_samples_per_class*2:
        raise ValueError(f"The number of samples per class in the pretest set must be at least {train_samples_per_class*2}.")

    # Generate filenames for the cache files using the parametersç
    multiplier_str = f"{train_samples_per_class}sxc_{pretrain_sample_multiplier}p_{test_sample_multiplier}t_{STS_sample_multiplier}STS"
    cache_file = dir_cache / f"{dataset}_{multiplier_str}_{pattern_type}_fold{fold_number}_rs{random_state}.npz"

    # If the cache file exists, load everything from there
    if cache_file.exists():

        log.info(f"Loading data from cache file {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        STS_tra, SCS_tra = data["STS_tra"], data["SCS_tra"]
        STS_pre, SCS_pre = data["STS_pre"], data["SCS_pre"]
        DM_tra, DM_pre = data["DM_tra"], data["DM_pre"]
        patterns = data["patterns"]

    else:

        # Generate the STSs
        STS_tra_samples = int(train_samples_per_class*n_classes)*STS_sample_multiplier
        STS_pre_samples = STS_tra_samples*pretrain_sample_multiplier + STS_tra_samples*test_sample_multiplier

        STS_tra, SCS_tra = compute_STS( # Generate train STS
            X_train, Y_train, fix_limits=True, STS_samples=STS_tra_samples+1, mode="random", random_state=random_state)
        STS_pre, SCS_pre = compute_STS( # Generate pretest STS
            X_pretest, Y_pretest, fix_limits=True, STS_samples=STS_pre_samples+1, mode="random", random_state=random_state)

        # Generate the patterns for the DMs
        if pattern_type == "medoids":
            log.info("Selecting the medoids from the train data")
            medoids, medoid_ids = compute_medoids(X_train, Y_train, distance_type="dtw")
            patterns = medoids
  
         # Generate the DMs
        DM_tra = compute_DM(STS_tra, patterns, rho_dfs, num_workers = num_workers)
        DM_pre = compute_DM(STS_pre, patterns, rho_dfs, num_workers = num_workers)

        # Remove the first sample from the STSs
        STS_tra = STS_tra[s_length:]
        STS_pre = STS_pre[s_length:]
        SCS_tra = SCS_tra[s_length:]
        SCS_pre = SCS_pre[s_length:]
        DM_tra = DM_tra[:,:,s_length:]
        DM_pre = DM_pre[:,:,s_length:]

        # Save the data to the cache file
        np.savez(cache_file, patterns=patterns,
            STS_tra=STS_tra, SCS_tra=SCS_tra, DM_tra=DM_tra,
            STS_pre=STS_pre, SCS_pre=SCS_pre, DM_pre=DM_pre)


    # Return the DataModule
    return FullDataModule()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def setup_trainer(
    directory: Path,
    version: str,
    epoch_max: int,
    stop_metric: str = "val_acc",
    mode: str = "max",
    ) -> tuple[Trainer, ModelCheckpoint]:

    """ Shared setup for the Trainer objects. """

    checkpoint = ModelCheckpoint(monitor=stop_metric, mode=mode)    
    trainer = Trainer(default_root_dir=directory,  accelerator="auto", devices="auto",
        # progress logs
        logger = [
            TensorBoardLogger(save_dir=directory, name="logs", version=version),
            CSVLogger(save_dir=directory, name="logs", version=version)
        ],
        callbacks=[
            # early stop the model         
            LearningRateMonitor(logging_interval='epoch'),  # learning rate logger
            checkpoint  # save best model version
            ],
        max_epochs=epoch_max,  deterministic = False, benchmark=True,
        log_every_n_steps=1, check_val_every_n_epoch=1
    )

    return trainer, checkpoint

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_model(
    directory: Path,
    label: str,
    epoch_max: int,
    dm: FullDataModule,
    target: str, 
    arch: type[LightningModule],
    approach: str="lstm",
    encoder: LightningModule = None,
    learning_rate: float = 1e-4,
    ) -> tuple[pd.DataFrame, WrapperModel, ModelCheckpoint]:

    if target == "cls":
        stop_metric, mode, metrics = "val_acc", "max", ["acc", "f1", "auroc"]
    elif target == "reg":
        stop_metric, mode, metrics = "val_mse", "min", ["mse", "r2"]

    results = pd.Series(dtype="object")
    frames: bool = arch.__frames__()

    # create the model
    if frames:
        model = WrapperModel(
                n_labels=dm.n_labels, 
                n_patterns=dm.n_patterns,
                l_patterns=dm.l_patterns,
                window_length=dm.window_length,
                lab_shifts=[0],
                arch=arch, 
                target=target,
                approach=approach,
                learning_rate=learning_rate)
    else:
        model = WrapperModel(
                n_labels=dm.n_labels, 
                n_patterns=1,
                l_patterns=1,
                window_length=dm.window_length,
                lab_shifts=[0],
                arch=arch,
                target=target, 
                approach=approach,
                learning_rate=learning_rate)
    
    # set encoder if one was passed
    if encoder is not None:
        model.encoder = encoder

    # train the model
    trainer, checkpoint = setup_trainer(directory=directory,  version=label,
        epoch_max=epoch_max, stop_metric=stop_metric, mode=mode)
    trainer.fit(model, datamodule=dm)

    # load best checkpoint
    model = model.load_from_checkpoint(checkpoint.best_model_path)

    # log val results
    val_results = trainer.validate(model, datamodule=dm)
    for m in metrics:
        results[f"{label}_val_{m}"] = val_results[0][f"val_{m}"]

    # log test results
    if dm.test:
        test_results = trainer.test(model, datamodule=dm)
        for m in metrics:
            results[f"{label}_test_{m}"] = test_results[0][f"test_{m}"]

    # load model info
    results["approach"] = approach
    results[f"{label}_best_model"] = checkpoint.best_model_path
    results[f"{label}_train_csv"] = str(directory  / "logs" / label / "metrics.csv")
    #results[f"{label}_nepochs"] = pd.read_csv(results[f"{label}_train_csv"])["epoch_train_acc"].count()
    results = results.to_frame().transpose().copy()

    return results, model, checkpoint

def base_results(dataset: str, fold_number: int, 
        arch: type[LightningModule], pretrained: bool, 
        batch_size: int, window_length: int, window_stride: int,
        random_state: int = 0) -> pd.DataFrame:
    
    """ Series template for the results. """

    df = pd.Series(dtype="object")
    df["dataset"], df["arch"], df["pretrained"]  = dataset, arch.__str__(), pretrained
    df["fold_number"], df["random_state"] = fold_number, random_state
    df["batch_size"], df["window_length"], df["window_stride"] = batch_size, window_length, window_stride
    df = df.to_frame().transpose().copy()

    return df