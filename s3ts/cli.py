#/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Command Line Interface (CLI) for the S3TS package.
"""

# package imports
from s3ts.data.acquisition import download_dataset
from s3ts.models.training import pretrain_encoder
from s3ts.models.training import train_model
from s3ts.data.setup import setup_pretrain_dm
from s3ts.data.setup import setup_train_dm 
from s3ts.data.setup import train_test_splits

# standard library
import multiprocessing as mp
from pathlib import Path
import logging as log
import argparse

# torch configuration
import torch
torch.set_float32_matmul_precision("medium")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='''Perform the experiments showcased in the article.''')

    parser.add_argument('--dataset', type=str, required=True,
        choices = ["GunPoint", "Plane", "CBF", "Fish", "Trace", "Chinatown", "OSULeaf", 
                    "PowerCons", "Car", "ECG200", "ECG5000", "ArrowHead"],
        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--mode', type=str, required=True, choices=['DF', 'TS'],
        help='Data representation (DF: Dissimilarity Frame, TS: Time Series)')

    parser.add_argument('--arch', type=str, required=True, choices=['CNN', 'ResNet'],
        help='Name of the architecture from which create the model')
    
    parser.add_argument('--use_pretrain', type=bool, default=False,
                        help='Use pretrained encoder or not (DF mode only)')

    parser.add_argument('--pretrain_mode', type=bool, default=False,
                        help='Switch between train and pretrain mode')

    parser.add_argument('--rho_dfs', type=float, default=0.1,
                        help='Value of the forgetting parameter')
    
    parser.add_argument('--window_length', type=int, default=10,
                        help='Window legth for the encoder')
    
    parser.add_argument('--stride_series', type=bool, default=False,
                        help='Stride or use the whole time series during pretrain')
    
    parser.add_argument('--window_time_stride', type=int, default=1,
                        help='Window time stride used for the encoder')
    
    parser.add_argument('--window_patt_stride', type=int, default=1,
                        help='Window pattern stride used for the encoder')
    
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
    
    parser.add_argument('--val_size', type=int, default=0.25,
                        help='Validation set size (as a fraction of the training set)')
    
    parser.add_argument('--max_epoch', type=int, default=60,
                        help='Maximum number of epochs for the training')
    
    parser.add_argument('--learning_rate', type=float, default=1e-04,
                        help='Value of the learning rate')
    
    parser.add_argument('--cv_rep', type=int, default=0,
                        help='Cross-validation repetition number')

    parser.add_argument('--random_state', type=int, default=0,
                        help='Global seed for the random number generators')

    parser.add_argument('--log_file', type=str, default="debug.log",
                        help='Directory for the results (CSVs)')
    
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

    # Training parameters
    exc: int = args.exc
    train_mult: int = args.train_mult
    train_strat_size: int = args.train_strat_size
    test_sts_length: int = args.test_sts_length
    pret_sts_length: int = args.pret_sts_length
    batch_size: int = args.batch_size
    val_size: float = args.val_size
    max_epoch: int = args.max_epochs
    learning_rate: float = args.learning_rate
    random_state: int = args.random_state
    cv_rep: int = args.cv_rep

    # Paths
    log_file: Path = Path(args.log_file)
    train_dir: Path = Path(args.train_dir)
    storage_dir: Path = Path(args.storage_dir)
    num_workers: int = args.num_workers

    # Print the arguments in a nice way for debugging
    log.info("Input Parameters:")
    for arg in vars(args):
        log.info(f"{arg}: {getattr(args, arg)}")

    # ~~~~~~~~~~~~ Sanity checks ~~~~~~~~~~~~

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
    enc_name1 = f"{dataset}_sl{pret_sts_length}_me{max_epoch}_rs{random_state}"
    enc_name2 = f"_ss{ss}_wl{window_length}_ts{window_time_stride}_ps{window_patt_stride}"
    encoder_path = storage_dir / "encoders" / (enc_name1 + enc_name2 + ".pt")

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
    if not use_pretrain:
        encoder_path = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Logging setup
    log.basicConfig(filename=log_file, level=log.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')

    # Download the dataset or load it from storage
    X, Y, medoids, medoid_idx = download_dataset(dataset=dataset, storage_dir=storage_dir)

    if pretrain_mode:

        directory = train_dir / "pretrain" / f"{arch}_{dataset}"

        dm = setup_pretrain_dm(X, Y, patterns=medoids, sts_length=pret_sts_length,
            rho_dfs=rho_dfs, batch_size=batch_size, val_size=val_size,
            window_length=window_length, stride_series=stride_series, 
            window_time_stride=window_time_stride, 
            window_patt_stride=window_patt_stride, 
            random_state=random_state,
            num_workers=num_workers)
        
        pretrain_encoder(dataset=dataset, repr="DF", arch=arch, dm=dm, directory=train_dir, 
            max_epoch=max_epoch, learning_rate=learning_rate, storage_dir=storage_dir)

    else:

        for j, (train_idx, test_idx) in enumerate(train_test_splits(X, Y, exc=exc, nreps=cv_rep+1, random_state=random_state)):
            if j == cv_rep:
                break

        dm = setup_train_dm(X=X, Y=Y, patterns=medoids,
                       train_idx=train_idx, test_idx=test_idx,
                       test_sts_length=test_sts_length,
                       train_strat_size=train_strat_size,
                       batch_size=batch_size, val_size=val_size,
                       train_mult=train_mult, rho_dfs=rho_dfs,
                       window_length=window_length, 
                       stride_series=stride_series,
                       window_time_stride=window_time_stride,
                       window_patt_stride=window_patt_stride,
                       random_state=random_state,
                       num_workers=num_workers)
        
        train_model(dataset=dataset, mode=mode, arch=arch,
                    dm=dm, pretrain=use_pretrain, fold_number=cv_rep,
                    max_epoch=max_epoch, learning_rate=learning_rate,)


    log.info("DONE!")