#!/usr/bin/env python

"""
Script that automates pretraining in a SBATCH queue.
"""

from s3ts.hooks import ATLAS_PRESET, sbatch_hook

DATASETS = ["CBF"]                                  # Datasets             
ARCHS = ["ResNet"]                                  # Architectures
WINDOW_LENGTHS: list[int] = [10]                    # Window length
WINDOW_TIME_STRIDES: list[int] = [1, 3, 5, 7]       # Window time stride
WINDOW_PATT_STRIDES: list[int] = [2, 3, 5]          # Window pattern stride

RHO_DFS: float = 0.1                # Memory parameter for DF
BATCH_SIZE: bool = 128              # Batch size
VAL_SIZE: float = 0.25              # Validation size
STS_LENGTH      = 1000              # Number of events for pretraining
MAX_EPOCH       = 60                # Number of epochs for pretraining 
LEARNING_RATE   = 1e-04             # Learning rate
RANDOM_STATE    = 0                 # Random state

for dataset in DATASETS:
    for arch in ARCHS:
        for wlen in WINDOW_LENGTHS:
            for wts in WINDOW_TIME_STRIDES:

                sbatch_hook(dataset=dataset, arch=arch, 
                    sts_length=STS_LENGTH, rho_dfs=RHO_DFS,
                    batch_size=BATCH_SIZE, val_size=VAL_SIZE,
                    max_epoch=MAX_EPOCH, learning_rate=LEARNING_RATE,
                    window_length=wlen, stride_series=True,
                    window_time_stride=wts, window_patt_stride=1,
                    random_state=RANDOM_STATE, **ATLAS_PRESET)
                
                sbatch_hook(dataset=dataset, arch=arch, 
                    sts_length=STS_LENGTH, rho_dfs=RHO_DFS,
                    batch_size=BATCH_SIZE, val_size=VAL_SIZE,
                    max_epoch=MAX_EPOCH, learning_rate=LEARNING_RATE,
                    window_length=wlen, stride_series=False,
                    window_time_stride=wts, window_patt_stride=1,
                    random_state=RANDOM_STATE, **ATLAS_PRESET)
                        
            for wps in WINDOW_PATT_STRIDES:

                sbatch_hook(dataset=dataset, arch=arch, 
                    sts_length=STS_LENGTH, rho_dfs=RHO_DFS,
                    batch_size=BATCH_SIZE, val_size=VAL_SIZE,
                    max_epoch=MAX_EPOCH, learning_rate=LEARNING_RATE,
                    window_length=wlen, stride_series=False,
                    window_time_stride=7, window_patt_stride=wps,
                    random_state=RANDOM_STATE, **ATLAS_PRESET)