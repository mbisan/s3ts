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
from s3ts.cli import main_loop

# standard library
import multiprocessing as mp
from pathlib import Path

import torch
torch.set_float32_matmul_precision("medium")

# Common settings
# ~~~~~~~~~~~~~~~~~~~~~~~
PRETRAIN_DF = True                  # Pretrain DF
TRAIN_DF = False                    # Train DF
TRAIN_TS = False                    # Train TS
# ~~~~~~~~~~~~~~~~~~~~~~~
DATASETS = [ # Datasets
    "CBF"#, "GunPoint", "Plane", "SyntheticControl"                                           
]                      
ARCHS = { # Architectures
    "DF": ["CNN"],# "ResNet"],
    "TS": ["RNN", "CNN", "ResNet"],
}
# ~~~~~~~~~~~~~~~~~~~~~~~
WINDOW_LENGTHS_TS: list[int] = [10, 30, 50, 70]     # Window length for TS                   
WINDOW_LENGTHS_DF: list[int] = [10]                 # Window length for DF
WINDOW_TIME_STRIDES: list[int] = [1, 3, 5, 7]       # Window time stride
WINDOW_PATT_STRIDES: list[int] = [2, 3, 5]          # Window pattern stride
# ~~~~~~~~~~~~~~~~~~~~~~~
RHO_DFS: float = 0.1                # Memory parameter for DF
BATCH_SIZE: bool = 128              # Batch size
VAL_SIZE: float = 0.25              # Validation size
NUM_ENCODER_FEATS: int = 32         # Number of encoder features
NUM_DECODER_FEATS: int = 64         # Number of decoder features
# ~~~~~~~~~~~~~~~~~~~~~~~
EVENTS_PER_CLASS = 32               # Number of events per class
TRAIN_EVENT_MULT = 4                # Training events multiplier
TRAIN_STRAT_SIZE = 2                # Training stratification size
TEST_STS_LENGTH = 200               # Number of events for testing
PRET_STS_LENGTH = 100               # Number of events for pretraining
# ~~~~~~~~~~~~~~~~~~~~~~~
MAX_EPOCHS_PRE: int = 20            # Pre-training epochs
MAX_EPOCHS_TRA: int = 120           # Training epochs
LEARNING_RATE: float = 1E-4         # Learning rate
# ~~~~~~~~~~~~~~~~~~~~~~~
LOG_FILE = None                     # Log file
RES_FNAME = "results.csv"           # Results filename
TRAIN_DIR = Path("training/")       # Training folder
STORAGE_DIR = Path("storage/")      # Cache folder
# ~~~~~~~~~~~~~~~~~~~~~~~
NUM_WORKERS = mp.cpu_count()//2     # Number of workers for the dataloaders
RANDOM_STATE = 0                    # Random state
CV_REPS = 5                         # Number of cross-validation repetitions
# ~~~~~~~~~~~~~~~~~~~~~~~
SHARED_ARGS = {"rho_dfs": RHO_DFS, "batch_size": BATCH_SIZE, "val_size": VAL_SIZE,
    "num_encoder_feats": NUM_ENCODER_FEATS, "num_decoder_feats": NUM_DECODER_FEATS,
    "exc": EVENTS_PER_CLASS, "learning_rate": LEARNING_RATE,
    "train_event_mult": TRAIN_EVENT_MULT, "train_strat_size": TRAIN_STRAT_SIZE,
    "test_sts_length": TEST_STS_LENGTH, "pret_sts_length": PRET_STS_LENGTH,
    "log_file": LOG_FILE, "res_fname": RES_FNAME, 
    "train_dir": TRAIN_DIR, "storage_dir": STORAGE_DIR,
    "num_workers": NUM_WORKERS, "random_state": RANDOM_STATE}

# Pretrain Loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if PRETRAIN_DF:
    for arch in ARCHS["DF"]:
        for dataset in DATASETS:
            for wlen in WINDOW_LENGTHS_DF:
                for wts in WINDOW_TIME_STRIDES:
                    # Full series
                    main_loop(dataset=dataset, mode="DF", arch=arch,
                        use_pretrain=False, pretrain_mode=True,
                        window_length=wlen, stride_series=False,
                        window_time_stride=wts, window_patt_stride=1,
                        max_epochs=MAX_EPOCHS_PRE, cv_rep=0, 
                        **SHARED_ARGS)
                    if wts != 1:
                        # Strided series
                        main_loop(dataset=dataset, mode="DF", arch=arch,
                            use_pretrain=False, pretrain_mode=True,
                            window_length=wlen, stride_series=True,
                            window_time_stride=wts, window_patt_stride=1,
                            max_epochs=MAX_EPOCHS_PRE, cv_rep=0, 
                            **SHARED_ARGS)
                for wps in WINDOW_PATT_STRIDES:
                    # Full series
                    main_loop(dataset=dataset, mode="DF", arch=arch,
                        use_pretrain=False, pretrain_mode=True,
                        window_length=wlen, stride_series=False,
                        window_time_stride=7, window_patt_stride=wps,
                        max_epochs=MAX_EPOCHS_PRE, cv_rep=0, 
                        **SHARED_ARGS)

# Training Loop for TS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if TRAIN_TS:
    mode = "TS"
    for cv_rep in range(CV_REPS):
        for arch in ARCHS[mode]:
            for dataset in DATASETS:

                # Check different window lengths
                for wlen in WINDOW_LENGTHS_TS:
                    main_loop(dataset=dataset, mode=mode, arch=arch,
                        use_pretrain=False, pretrain_mode=False,
                        window_length=wlen, stride_series=False,
                        window_time_stride=1, window_patt_stride=1,
                        max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                        **SHARED_ARGS)
                    
                # Check different event limits
                for exc in EVENTS_PER_CLASS:
                    main_loop(dataset=dataset, mode=mode, arch=arch,
                        use_pretrain=False, pretrain_mode=False,
                        window_length=WINDOW_LENGTHS_TS[0], stride_series=False,
                        window_time_stride=1, window_patt_stride=1,
                        max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                        exc=exc, **SHARED_ARGS)

# Training Loop for DF 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if TRAIN_DF:
    mode = "DF"
    for cv_rep in range(CV_REPS):
        for arch in ARCHS[mode]:
            for dataset in DATASETS:
                for wlen in WINDOW_LENGTHS_DF:
                    for wts in WINDOW_TIME_STRIDES:
                        # Full series
                        for use_pretrain in [False, True]:
                            main_loop(dataset=dataset, mode=mode, arch=arch,
                                use_pretrain=use_pretrain, pretrain_mode=False,
                                window_length=wlen, stride_series=False,
                                window_time_stride=wts, window_patt_stride=1,
                                max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                                **SHARED_ARGS)
                        if wts != 1:
                            # Strided series
                            for use_pretrain in [False, True]:
                                main_loop(dataset=dataset, mode=mode, arch=arch,
                                    use_pretrain=use_pretrain, pretrain_mode=False,
                                    window_length=wlen, stride_series=True,
                                    window_time_stride=wts, window_patt_stride=1,
                                    max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                                    **SHARED_ARGS)
                    for wps in WINDOW_PATT_STRIDES:
                        # Full series
                        for use_pretrain in [False, True]:
                            main_loop(dataset=dataset, mode=mode, arch=arch,
                                use_pretrain=use_pretrain, pretrain_mode=False,
                                window_length=wlen, stride_series=False,
                                window_time_stride=7, window_patt_stride=wps, 
                                max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                                **SHARED_ARGS)
