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
    "GunPoint", "Chinatown", "ECG200", "Crop", "CBF", "FordA", "DiatomSizeReduction"
    ]             
ARCHS = {                           # Architectures
    "DF": ["CNN", "ResNet"],
    #"TS": ["RNN", "CNN", "ResNet"])
    }
# ~~~~~~~~~~~~~~~~~~~~~~~
EXC: int = 32                       # Events per class (train)
TRAIN_EVENT_MULTIPLIER: int = 4     # Training sample multiplier
PRET_MULTIPLIER: int = 8            # Pre-training sample multiplier
TEST_MULTIPLIER: int = 4            # Test sample multiplier
# ~~~~~~~~~~~~~~~~~~~~~~~
RHO_DFS: float = 0.1                # Memory parameter for DF
PATT_TYPE: str = "medoids"          # Pattern type
BATCH_SIZE: bool = 128              # Batch size
VAL_SIZE: float = 0.4               # Validation size
WINDOW_LENGTH: int = 5              # Window length
STRIDE_SERIES: bool = False         # Stride the time series during pretrain
WINDOW_TIME_STRIDE: int = 1         # Window time stride
WINDOW_PATT_STRIDE: int = 1         # Window pattern stride
# ~~~~~~~~~~~~~~~~~~~~~~~
PRE_MAXEPOCH: int = 2               # Pre-training epochs
TRA_MAXEPOCH: int = 2               # Training epochs
LEARNING_RATE: float = 1E-4         # Learning rate
# ~~~~~~~~~~~~~~~~~~~~~~~
USE_CACHE = False                   # Use cache
CACHE_DIR = Path("cache/")          # Cache folder
TRAIN_DIR = Path("training/")       # Training folder
RESULTS_DIR = Path("results/")      # Results folder
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
    
for repr in ARCHS:
    for arch in ARCHS[repr]:
        for dataset in DATASETS:

            # Download dataset
            X, Y = download_dataset(dataset=dataset, cache_dir=CACHE_DIR)
        
            log.info(f"Current dataset: {dataset}")
            log.info(f"Current decoder: {arch} ({repr})")
    
            # Split data
            for j, (train_index, pretest_index) in enumerate(
                train_pretest_split(X, Y, exc=EXC, nreps=NREPS, random_state=RANDOM_STATE)):

                X_train, Y_train = X[train_index,:], Y[train_index]
                X_pretest, Y_pretest = X[pretest_index,:], Y[pretest_index]

                EXP(dataset=dataset, repr=repr, arch=arch,
                    X_train=X_train, X_pretest=X_pretest,
                    Y_train=Y_train, Y_pretest=Y_pretest,
                    fold_number=j, total_folds=NREPS, 
                    rho_dfs=RHO_DFS, pattern_type=PATT_TYPE, 
                    batch_size=BATCH_SIZE, val_size=VAL_SIZE,
                    window_length=WINDOW_LENGTH, stride_series=STRIDE_SERIES,
                    window_time_stride=WINDOW_TIME_STRIDE, window_patt_stride=WINDOW_PATT_STRIDE,
                    train_events_per_class=EXC, train_event_multiplier=TRAIN_EVENT_MULTIPLIER,
                    pret_event_multiplier=PRET_MULTIPLIER, test_event_multiplier=TEST_MULTIPLIER,
                    max_epoch_pre=PRE_MAXEPOCH, max_epoch_tra=TRA_MAXEPOCH,
                    learning_rate=LEARNING_RATE, random_state=RANDOM_STATE,
                    use_cache=USE_CACHE, cache_dir=CACHE_DIR, 
                    train_dir=TRAIN_DIR, results_dir=RESULTS_DIR)
