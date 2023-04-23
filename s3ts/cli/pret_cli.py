#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
    Command line interface for S3TS experiments. 

    This script is used to train all the models in the paper.
    It is possible to choose the experiment type (base, ratio, quant, stride)
    and the datasets to be used.
    The script will automatically train all the models for all the datasets.
    The results will be saved in the folder "results/".
    The training logs will be saved in the folder "training/".
    The datasets will be downloaded in the folder "cache/".
    The script will automatically create the folders if they do not exist.
    
    Usage:
        python main_cli.py --dataset <dataset> --mode <mode> --arch <arch> --exp <exp>
"""

# package imports
from s3ts.data.acquisition import download_dataset
from s3ts.models.training import pretrain_encoder
from s3ts.data.setup import setup_pretrain_dm 

# standard library
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

    parser.add_argument('--arch', type=str, required=True, choices=['CNN', 'ResNet'],
                        help='Name of the architecture from which create the model')

    parser.add_argument('--window_length', type=int, required=True,
                        help='Window legth for the encoder')
    
    parser.add_argument('--stride_series', type=bool, required=True,
                        help='Stride or use the whole time series during pretrain')
    
    parser.add_argument('--window_time_stride', type=int, required=True,
                        help='Window time stride used for the encoder')
    
    parser.add_argument('--window_patt_stride', type=int, required=True,
                        help='Window pattern stride used for the encoder')
    
    parser.add_argument('--sts_length', type=int, required=True,
                        help='Length of the STS used for pretraining')

    parser.add_argument('--rho_dfs', type=float, default=0.1,
                        help='Value of the forgetting parameter')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used during training')
    
    parser.add_argument('--val_size', type=int, default=0.25,
                        help='Validation set size (as a fraction of the training set)')
    
    parser.add_argument('--max_epoch', type=int, default=60,
                        help='Number of epochs to pretrain the networks')
    
    parser.add_argument('--learning_rate', type=float, default=1e-04,
                        help='Value of the learning rate')
    
    parser.add_argument('--random_state', type=int, default=0,
                        help='Global seed for the random number generators')

    parser.add_argument('--log_file', type=str, default="debug.log",
                        help='Directory for the results (CSVs)')
    
    parser.add_argument('--train_dir', type=str, default="training/",
                        help='Directory for the training files')
    
    parser.add_argument('--storage_dir', type=str, default="storage/",
                        help='Directory for the data storage')

    args = parser.parse_args()

    # Load the variables
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    arch: str = args.arch
    dataset: str = args.dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~
    rho_dfs: float = args.rho_dfs
    val_size: float = args.val_size
    sts_length: int = args.sts_length
    batch_size: int = args.batch_size
    window_length: int = args.window_length
    stride_series: bool = args.stride_series
    window_time_stride: int = args.window_time_stride
    window_patt_stride: int = args.window_patt_stride
    # ~~~~~~~~~~~~~~~~~~~~~~~
    max_epoch: int = args.max_epoch
    learning_rate: float = args.learning_rate
    # ~~~~~~~~~~~~~~~~~~~~~~~
    log_file: Path = Path(args.log_file)
    train_dir: Path = Path(args.train_dir)
    storage_dir: Path = Path(args.storage_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~
    random_state: int = args.random_state
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # logging setup
    log.basicConfig(filename=log_file, level=log.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')

    # load dataset
    X, Y, medoids, medoid_idx = download_dataset(dataset=dataset, storage_dir=storage_dir)

    directory = train_dir / "pretrain" / f"{arch}_{dataset}"

    log.info("Pretraining the encoder for:")
    log.info(f"Dataset: {dataset}")
    log.info(f"Achitecture: {arch}")
    log.info(f"Stride series: {stride_series}")
    log.info(f"Window length: {window_length}")
    log.info(f"Window time stride: {window_time_stride}")
    log.info(f"Window pattern stride: {window_patt_stride}")
    log.info(f"Stored in: {directory}")

    dm = setup_pretrain_dm(X, Y, patterns=medoids, sts_length=sts_length,
        rho_dfs=rho_dfs, batch_size=batch_size, val_size=val_size,
        window_length=window_length, stride_series=stride_series, 
        window_time_stride=window_time_stride, 
        window_patt_stride=window_patt_stride, 
        random_state=random_state)
    
    pretrain_encoder(dataset=dataset, repr="DF", arch=arch, dm=dm, directory=train_dir, 
        max_epoch=max_epoch, learning_rate=learning_rate, storage_dir=storage_dir)
                
    log.info("DONE!")