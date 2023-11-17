#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Command Line Interface (CLI) for the S3TS package. """

# package imports
from s3ts.models.training import run_model, save_results
from s3ts.data.acquisition import download_dataset
from s3ts.experiments.setup import setup_pretrain_dm
from s3ts.experiments.setup import setup_train_dm 
from s3ts.experiments.setup import train_test_splits

# standard library
import multiprocessing as mp
from pathlib import Path
import logging as log
import numpy as np
import argparse
import time, sys

# torch configuration
import torch
torch.set_float32_matmul_precision("medium")

from s3ts.experiments.synthetic

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='''Perform the experiments showcased in the article.''')

    parser.add_argument('--dataset', type=str, required=True,
        choices = ["ArrowHead", "CBF", "ECG200", "ECG5000", "GunPoint", "SyntheticControl", "Trace", "TwoLeadECG"],
        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--mode', type=str, required=True, choices=['ts', 'df', 'gf'],
        help='Data representation (df: dissimilarity frames, ts: time series)')

    parser.add_argument('--arch', type=str, required=True, choices=['nn', 'rnn', 'cnn', 'res', 'tcn', 'dfn'],
        help='Name of the architecture from which create the model')
    
    parser.add_argument('--use_pretrain', type=bool, action=argparse.BooleanOptionalAction, 
                        default=False, help='Use pretrained encoder or not (df/gf mode only)')

    parser.add_argument('--pretrain_mode', type=bool, action=argparse.BooleanOptionalAction,  
                        default=False, help='Switch between train and pretrain mode')

    parser.add_argument('--rho_dfs', type=float, default=0.1,
                        help='Value of the forgetting parameter for the DF representation')
    
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
                        help='Number of events per class')

    parser.add_argument('--train_strat_size', type=int, default=2,
                        help='Stratification size for the training STS')

    parser.add_argument('--train_event_mult', type=int, default=4,
                        help='Event multiplier for the training STS')
    
    parser.add_argument('--train_exc_limit', type=int, default=None,
                        help='Limit the number of available events per class (experimentation purposes)')

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
    
    parser.add_argument('--learning_rate', type=float, default=1E-4,
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
    train_exc_limit: int = args.train_exc_limit
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
        train_exc_limit=train_exc_limit,
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
