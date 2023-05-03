#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Common functions for the experiments. """

# in-package imports
from s3ts.data.oesm import compute_DM_optim
from s3ts.data.modules import DFDataModule
from s3ts.data.series import compute_STS

# standard library
import multiprocessing as mp
from pathlib import Path
import logging as log

# basics
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def train_test_splits(X: np.ndarray, Y: np.ndarray, 
        exc: int, nreps: int, random_state: int):

    """ Splits the dataset into train and pretest sets.
    Selects sxc events per class for the train set and the rest for the pretest set.
    
    Parameters
    ----------
    X : np.ndarray
        The time series dataset.
    Y : np.ndarray
        The labels of the time series dataset.
    sxc : int
        The number of events per class in the train set.
    nreps : int
        The number of different splits.
    random_state : int
        Random state for the RNG.
    """

    # Check the shape of the dataset and labels match
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of events in the dataset and labels must be the same.")

    # Check the number of events per class is not larger than the total number of events
    if exc > X.shape[0]:
        raise ValueError("The number of events per class cannot be larger than the total number of events.")
    
    # Check the number of events per class is not larger than the number of events per class
    if exc > np.unique(Y, return_counts=True)[1].min():
        raise ValueError("The number of events per class cannot be larger than the minimum number of events per class.")

    idx = np.arange(X.shape[0])
    rng = np.random.default_rng(random_state)

    for _ in range(nreps):
        
        train_idx = []
        for c in np.unique(Y):
            train_idx.append(rng.choice(idx, size=exc, p=(Y==c).astype(int)/sum(Y==c), replace=False))
        train_idx = np.concatenate(train_idx)
        test_idx = np.setdiff1d(idx, train_idx)

        yield train_idx, test_idx

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def setup_train_dm(
        X: np.ndarray, Y: np.ndarray, patterns: np.ndarray,
        train_idx: np.ndarray, test_idx: np.ndarray, 
        test_sts_length: int, 
        train_event_limit: int,
        train_strat_size: int,
        train_event_mult: int, 
        rho_dfs: float, 
        batch_size: int, val_size: float,
        window_length: int,
        window_time_stride: int, 
        window_patt_stride: int,
        random_state: int,
        num_workers: int = mp.cpu_count()//2,
        ) -> DFDataModule:

    """ Sets up the training DataModule."""

    # Get the train, test and medoid events
    X_train, Y_train = X[train_idx,:], Y[train_idx]
    X_test, Y_test = X[test_idx,:], Y[test_idx]

    # Validate the inputs
    event_length = X_train.shape[1]     # Get the length of the time series

    # Check there is the same numbe of classes in train and test
    if len(np.unique(Y_train)) != len(np.unique(Y_test)):
        raise ValueError("The number of classes in train and test must be the same.")

    # Check there is the same number of events in each class in train
    if len(np.unique(np.unique(Y_train, return_counts=True)[1])) != 1:
        raise ValueError("The number of events in each class in train must be the same.")

    # Check the number of events in each class in train is a multiple of the stratification size
    if len(Y_train)%train_strat_size != 0:
        raise ValueError("The number of events in each class in train must be a multiple of the stratification size.")

    STS_nev_train = len(train_idx)*train_event_mult
    STS_nev_test = test_sts_length

    log.info("Generating the train STS")
    STS_train, SCS_train = compute_STS(X_train, Y_train,        
        shift_limits=True, STS_events=STS_nev_train, 
        mode="stratified", event_strat_size=train_strat_size,
        random_state=random_state, add_first_event=True)
    
    log.info("Generating the test STS")
    STS_test, SCS_test = compute_STS(X_test, Y_test,                
        shift_limits=True, STS_events=STS_nev_test, mode="random", 
        random_state=random_state, add_first_event=True)

    log.info("Computing the train DM")
    DM_train = compute_DM_optim(STS_train, patterns, rho_dfs)

    log.info("Computing the test DM")
    DM_test = compute_DM_optim(STS_test, patterns, rho_dfs)

    # Remove the first sample from the STSs
    STS_train, STS_test = STS_train[event_length:], STS_test[event_length:]
    SCS_train, SCS_test = SCS_train[event_length:], SCS_test[event_length:]
    DM_train, DM_test = DM_train[:,:,event_length:], DM_test[:,:,event_length:]

    # Remove events according to train_event_limit
    limit_idx = event_length*train_event_limit
    STS_train, STS_test = STS_train[:limit_idx], STS_test[:limit_idx]
    SCS_train, SCS_test = SCS_train[:limit_idx], SCS_test[:limit_idx]
    DM_train, DM_test = DM_train[:,:,:limit_idx], DM_test[:,:,:limit_idx]


    # Return the DataModule
    return DFDataModule(
        STS_train=STS_train, SCS_train=SCS_train, DM_train=DM_train,
        STS_test=STS_test, SCS_test=SCS_test, DM_test=DM_test,
        event_length=event_length, patterns=patterns,
        batch_size=batch_size, val_size=val_size, 
        stride_series=False, window_length=window_length,
        window_time_stride=window_time_stride, 
        window_patt_stride=window_patt_stride,
        random_state=random_state, 
        num_workers=num_workers)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def setup_pretrain_dm(
        X: np.ndarray, Y: np.ndarray, patterns: np.ndarray, 
        sts_length: int, rho_dfs: float, 
        batch_size: int, val_size: float,
        window_length: int,
        window_time_stride: int, 
        window_patt_stride: int,
        random_state: int,
        stride_series: bool = False,
        num_workers: int = mp.cpu_count()//2,
        ) -> DFDataModule:

    """ Prepare the pretraining DataModule. """

    # Get the length of the time series
    event_length = X.shape[1]
    
    log.info("Generating the pretrain STS")
    STS_pret, SCS_pret = compute_STS(X, Y,                
        shift_limits=True, STS_events=sts_length, 
        mode="random", random_state=random_state, 
        add_first_event=True)

    log.info("Computing the pretrain DM")
    DM_pret = compute_DM_optim(STS_pret, patterns, rho_dfs)

    # Remove the first sample from the STSs
    STS_pret = STS_pret[event_length:]
    SCS_pret = SCS_pret[event_length:]
    DM_pret = DM_pret[:,:,event_length:]

    # Return the DataModule
    return DFDataModule(
        STS_train=STS_pret, SCS_train=SCS_pret, DM_train=DM_pret,
        STS_test=None, SCS_test=None, DM_test=None,
        event_length=event_length, patterns=patterns,
        batch_size=batch_size, val_size=val_size,
        stride_series=stride_series, 
        window_length=window_length,
        window_time_stride=window_time_stride, 
        window_patt_stride=window_patt_stride,
        random_state=random_state, 
        num_workers=num_workers)

