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

    parser.add_argument('--exc', type=int, default=16,
                        help='Number of samples per class')
    
    parser.add_argument('--train_mult', type=int, default=2,
                        help='Number of samples per class')
    
    parser.add_argument('--pret_mult', type=int, default=16,
                        help='Number of samples per class')

    parser.add_argument('--test_mult', type=int, default=2,
                        help='Number of samples per class')

    parser.add_argument('--rho_dfs', type=float, default=0.1,
                        help='Value of the forgetting parameter')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used during training')

    parser.add_argument('--window_length', type=int, default=5,
                        help='Window size used for training')
    
    parser.add_argument('--window_stride', type=int, default=1,
                        help='Window size used for training')
    
    parser.add_argument('--quant_intervals', type=int, default=5,
                        help='Number of quantiles used in pretraining')
    
    parser.add_argument('--quant_shift', type=float, default=0.0,
                        help='Label shift during pretraining')

    parser.add_argument('--pre_maxepoch', type=int, default=60,
                        help='Number of epochs to pretrain the networks')
    
    parser.add_argument('--tra_maxepoch', type=int, default=120,
                        help='Number of epochs to train the networks')
    
    parser.add_argument('--learning_rate', type=float, default=1e-04,
                        help='Value of the learning rate')
    
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of splits for K-fold validation')
    
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold from the K-fold.')

    parser.add_argument('--random_state', type=int, default=0,
                        help='Global seed for the random number generators')
    
    parser.add_argument('--use_cache', type=bool, default=False,
                        help='Use cached data if available')

    parser.add_argument('--dir_cache', type=str, default="cache/",
                        help='Directory for the cached data')
    
    parser.add_argument('--dir_train', type=str, default="training/",
                        help='Directory for the training files')
    
    parser.add_argument('--dir_results', type=str, default="results/",
                        help='Directory for the results (CSVs)')
    
    parser.add_argument('--log_file', type=str, default="debug.log",
                        help='Directory for the results (CSVs)')

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # sanity checks and variable selectors
    arch_dict = {"TS": ["RNN", "CNN", "ResNet"], "DF": ["CNN", "ResNet"]}
    if args.arch not in arch_dict[args.repr]:
        raise ValueError(f"Architecture {args.arch} not available for representation {args.repr}.")
    exp_dict = {"ratio": EXP_ratio}
    
    # get rest of variables
    dataset: str = args.dataset
    repr: str = args.repr
    arch: str = args.arch
    exp: function = exp_dict[args.exp]
    # ~~~~~~~~~~~~~~~~~~~~~~~
    exc: float = args.exc
    train_mult: float = args.train_mult
    pret_mult: float = args.pret_mult
    test_mult: float = args.test_mult
    # ~~~~~~~~~~~~~~~~~~~~~~~
    rho_dfs: float = args.rho_dfs
    batch_size: int = args.batch_size
    window_length: int = args.window_length
    window_time_stride: int = args.window_time_stride
    window_patt_stride: int = args.window_patt_stride
    # ~~~~~~~~~~~~~~~~~~~~~~~
    pre_maxepoch: int = args.pre_maxepoch
    tra_maxepoch: int = args.tra_maxepoch
    learning_rate: float = args.learning_rate
    # ~~~~~~~~~~~~~~~~~~~~~~~
    cache_dir: Path = Path(args.dir_cache)
    train_dir: Path = Path(args.dir_train)
    results_dir: Path = Path(args.dir_results)
    log_file: Path = Path(args.log_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~
    fold: int = args.fold
    nreps: int = args.nreps
    random_state: int = args.random_state
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # logging setup
    log.basicConfig(filename=log_file, level=log.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')

    # load dataset
    dataset: str = args.dataset
    X, Y, mapping = download_dataset(dataset_name=dataset, dir_cache=dir_cache)
    
    for j, (train_index, pretest_index) in enumerate(
        train_pretest_split(X, Y, exc=exc, nreps=nreps, random_state=random_state)):

        X_train, Y_train = X[train_index,:], Y[train_index]
        X_pretest, Y_pretest = X[pretest_index,:], Y[pretest_index]

        if fold is not None:
            if j != fold:
                continue

        X_train, Y_train = X[train_index,:], Y[train_index]
        X_test, Y_test = X[pretest_index,:], Y[pretest_index]
        
        EXP(dataset=dataset, repr=repr, arch=arch,
            X_train=X_train, X_pretest=X_pretest,
            Y_train=Y_train, Y_pretest=Y_pretest,
            fold_number=j, total_folds=nreps, rho_dfs=rho_dfs,
            batch_size=batch_size, window_length=window_length,
            window_time_stride=window_time_stride, window_pattern_stride=window_patt_stride,
            train_events_per_class=exc, train_event_multiplier=train_mult,
            pret_event_multiplier=pret_mult, test_event_multiplier=test_mult,
            max_epoch_pre=PRE_MAXEPOCH, max_epoch_tra=TRA_MAXEPOCH,
            learning_rate=learning_rate, random_state=RANDOM_STATE,
            use_cache=True, pattern_type="medoids",
            cache_dir=cache_dir, train_dir=train_dir, results_dir=results_dir)
