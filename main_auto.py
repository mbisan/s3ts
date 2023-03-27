#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
    Automatic training script for S3TS experiments. 
    
    This script is used to train all the models in the paper.
    It is possible to choose the experiment type (base, ratio, quant, stride)
    and the datasets to be used.
    The script will automatically train all the models for all the datasets.
    The results will be saved in the folder "results/".
    The training logs will be saved in the folder "training/".
    The datasets will be downloaded in the folder "cache/".
    The script will automatically create the folders if they do not exist.
"""

# data
from s3ts.experiments.common import train_pretest_split
from s3ts.data.acquisition import download_dataset

# experiments
from s3ts.experiments.ratio  import EXP_ratio

from pathlib import Path
import logging as log
import sys

log.basicConfig(
        stream=sys.stdout, 
        #filename="debug.log", 
        format="%(asctime)s [%(levelname)s] %(message)s", 
        level=log.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import torch
torch.set_float32_matmul_precision("medium")

# Common settings
# ~~~~~~~~~~~~~~~

EXP = "ratio"                       # Experiment
DATASETS = [                        # Datasets
    "GunPoint"
    ]             
ARCHS = {                           # Architectures
    "DF": {"CNN", "ResNet"},
    #"TS": ("RNN", "CNN", "ResNet")
    }
# ~~~~~~~~~~~~~~~~~~~~~~~
SXC: int = 16                       # Samples per class
PRET_MULTIPLIER: int = 16           # Pre-training sample multiplier
TEST_MULTIPLIER: int = 2            # Test sample multiplier
# ~~~~~~~~~~~~~~~~~~~~~~~
RHO_DFS: float = 0.1                # Memory parameter for DF
BATCH_SIZE: bool = 128              # Batch size
WINDOW_LENGTH: int = 5              # Window length
WINDOW_TIME_STRIDE: int = 1         # Window time stride
WINDOW_PATT_STRIDE: int = 1         # Window pattern stride
# ~~~~~~~~~~~~~~~~~~~~~~~
PRE_MAXEPOCH: int = 60              # Pre-training epochs
TRA_MAXEPOCH: int = 120             # Training epochs
LEARNING_RATE: float = 1E-4
# ~~~~~~~~~~~~~~~~~~~~~~~
DIR_CACHE = Path("cache/")          # Cache folder
DIR_TRAIN = Path("training/")       # Training folder
DIR_RESULTS = Path("results/")      # Results folder
# ~~~~~~~~~~~~~~~~~~~~~~~
NREPS = 5                                   # Number of repetitions
RANDOM_STATE = 0                            # Random state

# Choose experiment functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

exp_dict = {
    "ratio": EXP_ratio
}
EXP = exp_dict[EXP]

# Iterate over all the combinations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
for mode in ARCHS:
    for arch in ARCHS[mode]:
        for dataset in DATASETS:

            # Download dataset
            X, Y = download_dataset(dataset=dataset, dir_cache=DIR_CACHE)
        
            log.info(f"Current dataset: {dataset}")
            log.info(f"Current decoder: {arch} ({mode})")
    
            # Split data
            for j, (train_index, pretest_index) in enumerate(
                train_pretest_split(X, Y, sxc=SXC, nreps=NREPS, random_state=RANDOM_STATE)):

                X_train, Y_train = X[train_index,:], Y[train_index]
                X_pretest, Y_pretest = X[pretest_index,:], Y[pretest_index]

                EXP(dataset=dataset, repr=mode, arch=arch,
                    X_train=X_train, X_pretest=X_pretest,
                    Y_train=Y_train, Y_pretest=Y_pretest,
                    fold_number=j, total_folds=NREPS, rho_dfs=RHO_DFS,
                    batch_size=BATCH_SIZE, window_length=WINDOW_LENGTH,
                    window_time_stride=WINDOW_TIME_STRIDE, window_pattern_stride=WINDOW_PATT_STRIDE,
                    train_samples_per_class=SXC, train_sample_multiplier=PRET_MULTIPLIER,
                    pret_sample_multiplier=PRET_MULTIPLIER, test_sample_multiplier=TEST_MULTIPLIER,
                    max_epoch_pre=PRE_MAXEPOCH, max_epoch_tra=TRA_MAXEPOCH,
                    learning_rate=LEARNING_RATE, random_state=RANDOM_STATE,
                    use_cache=True, pattern_type="medoids",
                    cache_dir=DIR_CACHE, train_dir=DIR_TRAIN, results_dir=DIR_RESULTS)
