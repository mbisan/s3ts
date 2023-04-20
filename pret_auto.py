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
from s3ts.experiments.common import setup_pretrain_dm, pretrain_encoder
from s3ts.data.acquisition import download_dataset

# standard library
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

DATASETS = ["CBF"]                                  # Datasets             
ARCHS = {"DF": ["CNN"]}                             # Architectures
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
USE_CACHE = True                    # Use cache
TRAIN_DIR = Path("training/")       # Training folder
STORAGE_DIR = Path("storage/")      # Cache folder
# ~~~~~~~~~~~~~~~~~~~~~~~
NREPS = 5                           # Number of repetitions
RANDOM_STATE = 0                    # Random state

# Iterate over all the combinations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for repr in ARCHS:
    for arch in ARCHS[repr]:
        for dataset in DATASETS:

            # Download dataset
            X, Y, medoids, medoid_idx = download_dataset(dataset=dataset, storage_dir=STORAGE_DIR)
        
            log.info(f"Current dataset: {dataset}")
            log.info(f"Current decoder: {arch} ({repr})")
            
            train_dir = TRAIN_DIR / "pretrain" / f"{arch}_{dataset}"
    
            dm = setup_pretrain_dm(X, Y, patterns=medoids, sts_length=STS_PRET_EVENTS,
                rho_dfs=RHO_DFS, batch_size=BATCH_SIZE, val_size=VAL_SIZE,
                window_length=15, stride_series=False, window_time_stride=5, 
                window_patt_stride=1, random_state=RANDOM_STATE)

            for wlen in WINDOW_LENGTHS:

                dm.update_properties(window_length=wlen)

                for wts in WINDOW_TIME_STRIDES:

                    dm.update_properties(window_time_stride=wts, stride_series=True)

                    log.info(f"Current window length: {dm.window_length}")
                    log.info(f"Current window time stride: {dm.window_time_stride}")
                    log.info(f"Current window pattern stride: {dm.window_patt_stride}")

                    # Pretrain encoder
                    pretrain_encoder(dataset=dataset, repr=repr, arch=arch, dm=dm, directory=train_dir, 
                        max_epoch=MAXEPOCH, learning_rate=LEARNING_RATE, storage_dir=STORAGE_DIR)
                
                for wps in WINDOW_PATT_STRIDES:

                    dm.update_properties(window_patt_stride=wps)

                    log.info(f"Current window length: {wlen}")
                    log.info(f"Current window time stride: {wts}")
                    log.info(f"Current window pattern stride: {wps}")

                    # Pretrain encoder
                    pretrain_encoder(dataset=dataset, repr=repr, arch=arch, dm=dm, directory=train_dir, 
                        max_epoch=MAXEPOCH, learning_rate=LEARNING_RATE, storage_dir=STORAGE_DIR)
                    

