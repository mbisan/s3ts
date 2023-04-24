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

# package imports
from s3ts.hooks import python_hook

# standard library
from pathlib import Path
import logging as log
import sys

import torch
torch.set_float32_matmul_precision("medium")

# Common settings
# ~~~~~~~~~~~~~~~

DATASETS = ["CBF"]                                  # Datasets             
ARCHS = ["CNN", "ResNet"]                           # Architectures
WINDOW_LENGTHS: list[int] = [5,10,15]               # Window length
WINDOW_TIME_STRIDES: list[int] = [1, 3, 5]          # Window time stride
WINDOW_PATT_STRIDES: list[int] = [1, 2, 3, 5]       # Window pattern stride

# ~~~~~~~~~~~~~~~~~~~~~~~

RHO_DFS: float = 0.1                # Memory parameter for DF
BATCH_SIZE: bool = 128              # Batch size
VAL_SIZE: float = 0.25              # Validation size
STS_PRET_EVENTS = 1000              # Number of events for pretraining

# ~~~~~~~~~~~~~~~~~~~~~~~
MAXEPOCH: int = 60                  # Pre-training epochs
LEARNING_RATE: float = 1E-4         # Learning rate
# ~~~~~~~~~~~~~~~~~~~~~~~
TRAIN_DIR = Path("training/")       # Training folder
STORAGE_DIR = Path("storage/")      # Cache folder
# ~~~~~~~~~~~~~~~~~~~~~~~
RANDOM_STATE = 0                    # Random state

log.basicConfig(stream=sys.stdout, #filename="debug.log", 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    level=log.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Iterate over all the combinations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for arch in ARCHS:
    for dataset in DATASETS:

        train_dir = TRAIN_DIR / "pretrain" / f"{arch}_{dataset}"

        for wlen in WINDOW_LENGTHS:

            for wts in WINDOW_TIME_STRIDES:
            
                python_hook(dataset=dataset, mode="DF", arch=arch,
                    use_pretrain=False, pretrain_mode=True, rho_dfs=RHO_DFS, 
                    window_length=10, stride_series=True,
                    window_time_stride=5, window_patt_stride=1,
                    pret_sts_length=STS_PRET_EVENTS,
                    batch_size=BATCH_SIZE,
                    val_size=VAL_SIZE,
                    max_epoch=MAXEPOCH,
                    learning_rate=LEARNING_RATE,
                    random_state=RANDOM_STATE)
                
                python_hook(dataset=dataset, mode="DF", arch=arch,
                    use_pretrain=False, pretrain_mode=True, rho_dfs=RHO_DFS, 
                    window_length=10, stride_series=True,
                    window_time_stride=5, window_patt_stride=1,
                    pret_sts_length=STS_PRET_EVENTS,
                    batch_size=BATCH_SIZE,
                    val_size=VAL_SIZE,
                    max_epoch=MAXEPOCH,
                    learning_rate=LEARNING_RATE,
                    random_state=RANDOM_STATE)

            for wps in WINDOW_PATT_STRIDES:

                python_hook(dataset=dataset, mode="DF", arch=arch,
                    use_pretrain=False, pretrain_mode=True, rho_dfs=RHO_DFS, 
                    window_length=10, stride_series=True,
                    window_time_stride=5, window_patt_stride=1,
                    pret_sts_length=STS_PRET_EVENTS,
                    batch_size=BATCH_SIZE,
                    val_size=VAL_SIZE,
                    max_epoch=MAXEPOCH,
                    learning_rate=LEARNING_RATE,
                    random_state=RANDOM_STATE)
