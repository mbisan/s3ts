#/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Command Line Interface (CLI) for the S3TS package.
"""

# package imports
from s3ts.models.training import train_model, save_results
from s3ts.data.acquisition import download_dataset
from s3ts.data.setup import setup_pretrain_dm
from s3ts.data.setup import setup_train_dm 
from s3ts.data.setup import train_test_splits

# standard library
import multiprocessing as mp
from pathlib import Path
import logging as log
import argparse
import time, sys

# torch configuration
import torch
torch.set_float32_matmul_precision("medium")

def main_loop(
        dataset: str,
        mode: str,
        arch: str,
        use_pretrain: bool,
        pretrain_mode: bool,
        # Model parameters
        rho_dfs: float,
        window_length: int,
        stride_series: bool,
        window_time_stride: int,
        window_patt_stride: int,
        num_encoder_feats: int,
        num_decoder_feats: int,
        # Training parameters
        exc: int,
        train_event_mult: int,
        train_strat_size: int,
        test_sts_length: int,
        pret_sts_length: int,
        batch_size: int,
        val_size: float,
        max_epochs: int,
        learning_rate: float,
        random_state: int,
        cv_rep: int,
        # Paths
        log_file: Path,
        res_fname: str,
        train_dir: Path,
        storage_dir: Path,
        num_workers: int,
        ):
    
    # ~~~~~~~~~~~ Create folders ~~~~~~~~~~~~

    start_time = time.perf_counter()

    for fold in ["datasets", "results", "encoders"]:
        path = storage_dir / fold
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    # Logging setup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if log_file is not None:
        log.basicConfig(filename=log_file, level=log.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        log.basicConfig(stream=sys.stdout, level=log.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S')

    # ~~~~~~~~~~~~ Sanity checks ~~~~~~~~~~~~

    log.info("Performing sanity checks...")

    # Check all window parameters are positive integers
    for val in [window_length, window_time_stride, window_patt_stride]:
        if val < 1 or not isinstance(val, int):
            raise ValueError("Window paramters must be positive integers.")
    
    # Check mode is 'DF' if use_pretrain is True
    if use_pretrain and mode != "DF" or pretrain_mode and mode != "DF":
        raise ValueError("Pretraining is only available for DF mode.")

    # Check pretrain_mode and use_pretrain are not both True
    if use_pretrain and pretrain_mode:
        raise ValueError("'pretrain_mode' is a previous step to 'use_pretrain', so they cannot be both True.")

    # Get the path to the encoder
    ss = 1 if stride_series else 0
    enc_name = f"{dataset}_{arch}_wl{window_length}_ts{window_time_stride}_ps{window_patt_stride}_ss{ss}"
    encoder_path = storage_dir / "encoders" / (enc_name + ".pt")

    if use_pretrain or pretrain_mode:
        log.info(f"encoder_path: {encoder_path}")

    # If not in pretrain_mode and use_pretrain, check the encoder exists
    if use_pretrain and not pretrain_mode:
        if not encoder_path.exists():
            raise ValueError("Encoder not found. Please run pretrain mode first.")
    
    # If pretrain_mode, check the encoder does not exist already
    if pretrain_mode:
        if encoder_path.exists():
            raise ValueError("Encoder already exists. Please delete it before running pretrain mode.")
    
    # If use_pretrain is False, set encoder_path to None
    if not pretrain_mode and not use_pretrain:
        encoder_path = None
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Download the dataset or load it from storage
    X, Y, medoids, medoid_idx = download_dataset(dataset=dataset, storage_dir=storage_dir)

    if pretrain_mode:
        # Get directory and version
        directory = train_dir / "pretrain" / f"{dataset}_{arch}"
        version = f"wlen{window_length}_stride{ss}" +\
            f"_wtst{window_time_stride}_wpst{window_patt_stride}" +\
            f"_val{val_size}_me{max_epochs}_bs{batch_size}" +\
            f"_stsl{pret_sts_length}_lr{learning_rate}_rs{random_state}"
        # Setup the data module
        dm = setup_pretrain_dm(X, Y, patterns=medoids, 
            sts_length=pret_sts_length,rho_dfs=rho_dfs, 
            batch_size=batch_size, val_size=val_size,
            window_length=window_length, 
            stride_series=stride_series, 
            window_time_stride=window_time_stride, 
            window_patt_stride=window_patt_stride, 
            random_state=random_state,
            num_workers=num_workers)
    else:
        # Get the train and test idx for the current CV repetition
        for j, (train_idx, test_idx) in enumerate(train_test_splits(X, Y, exc=exc, nreps=cv_rep+1, random_state=random_state)):
            if j == cv_rep:
                break
        # Get directory and version
        directory = train_dir / "finetune" / f"{dataset}_{mode}_{arch}"
        # TODO: add option to reduce dm available events (ratio)
        if mode == "DF":
            version = f"wlen{window_length}_stride{ss}" +\
                f"_wtst{window_time_stride}_wpst{window_patt_stride}" +\
                f"_val{val_size}_me{max_epochs}_bs{batch_size}" +\
                f"_lr{learning_rate}_rs{random_state}"
        else:
            version = f"wlen{window_length}" +\
                f"_val{val_size}_me{max_epochs}_bs{batch_size}" +\
                f"_lr{learning_rate}_rs{random_state}"
            
        # Setup the data module
        dm = setup_train_dm(X=X, Y=Y, patterns=medoids,
            train_idx=train_idx, test_idx=test_idx,
            test_sts_length=test_sts_length,
            train_event_mult=train_event_mult,
            train_strat_size=train_strat_size,
            batch_size=batch_size, val_size=val_size,               
            rho_dfs=rho_dfs, window_length=window_length, 
            window_time_stride=window_time_stride,
            window_patt_stride=window_patt_stride,
            random_state=random_state,
            num_workers=num_workers)
    
    dm_time =time.perf_counter()
    
    # Train the model
    data, model = train_model(
        pretrain_mode=pretrain_mode, version=version,
        dataset=dataset, mode=mode, arch=arch, dm=dm, 
        directory=directory, max_epochs=max_epochs,
        learning_rate=learning_rate, encoder_path=encoder_path,
        num_encoder_feats=num_encoder_feats,
        num_decoder_feats=num_decoder_feats,
        random_state=random_state, cv_rep=cv_rep)
    
    train_time = time.perf_counter()
    
    if not pretrain_mode:
        data["train_strat_size"] = train_strat_size
        data["train_event_mult"] = train_event_mult

    # Log times
    data["time_dm"] = dm_time - start_time
    data["time_train"] = train_time - dm_time
    data["time_total"] = train_time - start_time

    # Save the results
    save_results(data, res_fname=res_fname, storage_dir=storage_dir)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='''Perform the experiments showcased in the article.''')

    parser.add_argument('--dataset', type=str, required=True,
        choices = ["GunPoint", "Plane", "CBF", "Fish", "Trace", "Chinatown", "OSULeaf", 
                    "PowerCons", "Car", "ECG200", "ArrowHead"],
        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--mode', type=str, required=True, choices=['DF', 'TS'],
        help='Data representation (DF: Dissimilarity Frame, TS: Time Series)')

    parser.add_argument('--arch', type=str, required=True, choices=['CNN', 'ResNet'],
        help='Name of the architecture from which create the model')
    
    parser.add_argument('--use_pretrain', type=bool, action=argparse.BooleanOptionalAction, 
                        default=False, help='Use pretrained encoder or not (DF mode only)')

    parser.add_argument('--pretrain_mode', type=bool, action=argparse.BooleanOptionalAction,  
                        default=False, help='Switch between train and pretrain mode')

    parser.add_argument('--rho_dfs', type=float, default=0.1,
                        help='Value of the forgetting parameter')
    
    parser.add_argument('--window_length', type=int, default=10,
                        help='Window legth for the encoder')
    
    parser.add_argument('--stride_series', type=bool, action=argparse.BooleanOptionalAction,  
                        default=False, help='Stride the time series during pretrain')

    parser.add_argument('--window_time_stride', type=int, default=1,
                        help='Window time stride used for the encoder')
    
    parser.add_argument('--window_patt_stride', type=int, default=1,
                        help='Window pattern stride used for the encoder')
    
    parser.add_argument('--num_encoder_feats', type=int, default=32,
                        help='Number of features used for the encoder.')
    
    parser.add_argument('--num_decoder_feats', type=int, default=64,
                        help='Number of features used for the encoder.')
    
    parser.add_argument('--exc', type=int, default=16,
                        help='Number of samples per class')

    parser.add_argument('--train_strat_size', type=int, default=2,
                        help='Stratification size for the training set')

    parser.add_argument('--train_event_mult', type=int, default=4,
                        help='Number of samples per class')

    parser.add_argument('--test_sts_length', type=int, default=200,
                        help='Length of the STS used for testing')
    
    parser.add_argument('--pret_sts_length', type=int, default=1000,
                        help='Length of the STS used for pretraining')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used during training')
    
    parser.add_argument('--val_size', type=float, default=0.25,
                        help='Validation set size (as a fraction of the training set)')
    
    parser.add_argument('--max_epochs', type=int, default=60,
                        help='Maximum number of epochs for the training')
    
    parser.add_argument('--learning_rate', type=float, default=1e-04,
                        help='Value of the learning rate')
    
    parser.add_argument('--cv_rep', type=int, default=0,
                        help='Cross-validation repetition number')

    parser.add_argument('--random_state', type=int, default=0,
                        help='Global seed for the random number generators')
    
    parser.add_argument('--log_file', type=str, default="debug.log",
                        help='Log file for the training')
    
    parser.add_argument('--res_fname', type=str, default="results.csv",
                        help='Results file for the training')
    
    parser.add_argument('--train_dir', type=str, default="training/",
                        help='Directory for the training files')
    
    parser.add_argument('--storage_dir', type=str, default="storage/",
                        help='Directory for the data storage')
    
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count()//2,
                        help='Number of workers for the data loaders')

    args = parser.parse_args()
   
    # ~~~~~~~~~~~~ Put the arguments in variables ~~~~~~~~~~~~
    
    # Basic Information
    dataset: str = args.dataset
    mode: str = args.mode
    arch: str = args.arch
    use_pretrain: bool = args.use_pretrain
    pretrain_mode: bool = args.pretrain_mode
    
    # Model parameters
    rho_dfs: float = args.rho_dfs
    window_length: int = args.window_length
    stride_series: bool = args.stride_series
    window_time_stride: int = args.window_time_stride
    window_patt_stride: int = args.window_patt_stride
    num_encoder_feats: int = args.num_encoder_feats
    num_decoder_feats: int = args.num_decoder_feats

    # Training parameters
    exc: int = args.exc
    train_event_mult: int = args.train_event_mult
    train_strat_size: int = args.train_strat_size
    test_sts_length: int = args.test_sts_length
    pret_sts_length: int = args.pret_sts_length
    batch_size: int = args.batch_size
    val_size: float = args.val_size
    max_epochs: int = args.max_epochs
    learning_rate: float = args.learning_rate
    random_state: int = args.random_state
    cv_rep: int = args.cv_rep

    # Paths
    log_file: Path = Path(args.log_file)
    res_fname: Path = Path(args.res_fname)
    train_dir: Path = Path(args.train_dir)
    storage_dir: Path = Path(args.storage_dir)
    num_workers: int = args.num_workers

    # Print the arguments in a nice way for debugging
    log.info("Input Parameters:")
    for arg in vars(args):
        log.info(f"{arg}: {getattr(args, arg)}")

    # ~~~~~~~~~~~~ Launch the training loop ~~~~~~~~~~~~
    main_loop(
        dataset=dataset,
        mode=mode,
        arch=arch,
        use_pretrain=use_pretrain,
        pretrain_mode=pretrain_mode,
        rho_dfs=rho_dfs,
        window_length=window_length,
        stride_series=stride_series,
        window_time_stride=window_time_stride,
        window_patt_stride=window_patt_stride,
        num_encoder_feats=num_encoder_feats,
        num_decoder_feats=num_decoder_feats,
        exc=exc,
        train_event_mult=train_event_mult,
        train_strat_size=train_strat_size,
        test_sts_length=test_sts_length,
        pret_sts_length=pret_sts_length,
        batch_size=batch_size,
        val_size=val_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        random_state=random_state,
        cv_rep=cv_rep,
        log_file=log_file,
        res_fname=res_fname,
        train_dir=train_dir,
        storage_dir=storage_dir,
        num_workers=num_workers)
