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
from s3ts.experiments.common import train_pretest_split
from s3ts.experiments.ratio  import EXP_ratio

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
                        choices = ["GunPoint", "Coffee", "PowerCons", "Plane", "CBF", 
                                   "ECG200", "Trace", "SyntheticControl", "Chinatown"],
                        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--repr', type=str, required=True, choices=['DF', 'TS'],
                        help='Data representation (DF: Dissimilarity Frame, TS: Time Series)')

    parser.add_argument('--arch', type=str, required=True, choices=['RNN', 'CNN', 'ResNet'],
                        help='Name of the architecture from which create the model')
    
    parser.add_argument('--exp', type=str, required=True, choices=['ratio'],
                        help='Name of the architecture from which create the model')

    parser.add_argument('--exc', type=int, default=32,
                        help='Number of samples per class')
    
    parser.add_argument('--train_mult', type=int, default=4,
                        help='Number of samples per class')
    
    parser.add_argument('--pret_mult', type=int, default=8,
                        help='Number of samples per class')

    parser.add_argument('--test_mult', type=int, default=4,
                        help='Number of samples per class')

    parser.add_argument('--rho_dfs', type=float, default=0.1,
                        help='Value of the forgetting parameter')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used during training')
    
    parser.add_argument('--val_size', type=int, default=0.4,
                        help='Validation set size (as a fraction of the training set)')
    
    parser.add_argument('--pattern_type', type=str, choices=['medoids'], default='medoids',
                        help='Patterns to be used for generating the DFs')

    parser.add_argument('--window_length', type=int, default=5,
                        help='Window size used for training')
    
    parser.add_argument('--stride_series', type=bool, default=False,
                        help='Stride or use the whole time series during pretrain')
    
    parser.add_argument('--window_time_stride', type=int, default=1,
                        help='Window size used for training')
    
    parser.add_argument('--window_patt_stride', type=int, default=1,
                        help='Window size used for training')

    parser.add_argument('--max_epoch_pre', type=int, default=60,
                        help='Number of epochs to pretrain the networks')
    
    parser.add_argument('--max_epoch_tra', type=int, default=120,
                        help='Number of epochs to train the networks')
    
    parser.add_argument('--learning_rate', type=float, default=1e-04,
                        help='Value of the learning rate')
    
    parser.add_argument('--rep', type=int, default=None,
                        help='Specific fold from the K-fold.')

    parser.add_argument('--nreps', type=int, default=5,
                        help='Number of splits for K-fold validation')
    
    parser.add_argument('--random_state', type=int, default=0,
                        help='Global seed for the random number generators')
    
    parser.add_argument('--use_cache', type=bool, default=True,
                        help='Use cache to store DFs and TSs')

    parser.add_argument('--dir_cache', type=str, default="cache/",
                        help='Directory for the cached data')
    
    parser.add_argument('--dir_train', type=str, default="training/",
                        help='Directory for the training files')
    
    parser.add_argument('--dir_results', type=str, default="results/",
                        help='Directory for the results (CSVs)')
    
    parser.add_argument('--log_file', type=str, default="debug.log",
                        help='Directory for the results (CSVs)')

    args = parser.parse_args()

    # Parameter checks
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Check if the architecture is available for the representation
    arch_dict = {"TS": ["RNN", "CNN", "ResNet"], "DF": ["CNN", "ResNet"]}
    if args.arch not in arch_dict[args.repr]:
        raise ValueError(f"Architecture {args.arch} not available for representation {args.repr}.")
    
    # Check if the experiment is available
    exp_dict = {"ratio": EXP_ratio}
    if args.exp not in exp_dict.keys():
        raise ValueError(f"Experiment {args.exp} not available.")
    
    # Check rep is in range [0, nreps)
    if args.rep is not None:
        if args.rep < 0 or args.rep >= args.nreps:
            raise ValueError(f"rep must be in range [0, nreps).")
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # get rest of variables
    repr: str = args.repr
    arch: str = args.arch
    exp = exp_dict[args.exp]
    dataset: str = args.dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~
    exc: float = args.exc
    train_mult: float = args.train_mult
    pret_mult: float = args.pret_mult
    test_mult: float = args.test_mult
    # ~~~~~~~~~~~~~~~~~~~~~~~
    rho_dfs: float = args.rho_dfs
    batch_size: int = args.batch_size
    val_size: float = args.val_size
    pattern_type: str = args.pattern_type
    window_length: int = args.window_length
    stride_series: bool = args.stride_series
    window_time_stride: int = args.window_time_stride
    window_patt_stride: int = args.window_patt_stride
    # ~~~~~~~~~~~~~~~~~~~~~~~
    max_epoch_pre: int = args.max_epoch_pre
    max_epoch_tra: int = args.max_epoch_tra
    learning_rate: float = args.learning_rate
    # ~~~~~~~~~~~~~~~~~~~~~~~
    use_cache: bool = args.use_cache
    cache_dir: Path = Path(args.dir_cache)
    train_dir: Path = Path(args.dir_train)
    results_dir: Path = Path(args.dir_results)
    log_file: Path = Path(args.log_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~
    rep: int = args.rep
    nreps: int = args.nreps
    random_state: int = args.random_state
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # logging setup
    log.basicConfig(filename=log_file, level=log.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')

    # load dataset
    X, Y = download_dataset(dataset=dataset, cache_dir=cache_dir)
    
    for j, (train_index, pretest_index) in enumerate(
        train_pretest_split(X, Y, exc=exc, nreps=nreps, random_state=random_state)):

        X_train, Y_train = X[train_index,:], Y[train_index]
        X_pretest, Y_pretest = X[pretest_index,:], Y[pretest_index]

        if rep is not None:
            if j != rep:
                continue

        X_train, Y_train = X[train_index,:], Y[train_index]
        X_test, Y_test = X[pretest_index,:], Y[pretest_index]
        
        exp(dataset=dataset, repr=repr, arch=arch,
            X_train=X_train, X_pretest=X_pretest,
            Y_train=Y_train, Y_pretest=Y_pretest,
            fold_number=j, total_folds=nreps, rho_dfs=rho_dfs,
            batch_size=batch_size, val_size=val_size,
            window_length=window_length, stride_series=stride_series,
            window_time_stride=window_time_stride, window_pattern_stride=window_patt_stride,
            train_events_per_class=exc, train_event_multiplier=train_mult,
            pret_event_multiplier=pret_mult, test_event_multiplier=test_mult,
            max_epoch_pre=max_epoch_pre, max_epoch_tra=max_epoch_tra,
            learning_rate=learning_rate, random_state=random_state,
            use_cache=use_cache, pattern_type=pattern_type,
            cache_dir=cache_dir, train_dir=train_dir, results_dir=results_dir)
