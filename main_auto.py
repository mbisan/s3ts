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
from s3ts.data.tasks.download import download_dataset
from sklearn.model_selection import StratifiedKFold

# architectures
from s3ts.models.encoders.frames.ResNet import ResNet_DFS
from s3ts.models.encoders.frames.CNN import CNN_DFS

from s3ts.models.encoders.series.ResNet import ResNet_TS
from s3ts.models.encoders.series.CNN import CNN_TS
from s3ts.models.encoders.series.RNN import RNN_TS

# experiments
from s3ts.experiments.base import EXP_base
from s3ts.experiments.ratio  import EXP_ratio
from s3ts.experiments.quant  import EXP_quant
from s3ts.experiments.stride import EXP_stride

from itertools import product
from pathlib import Path
import logging as log

log.basicConfig(filename="debug.log", level=log.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')

import torch
torch.set_float32_matmul_precision("medium")

# SETTINGS
# =================================

EXP = "base"
DATASETS = ["GunPoint", "ECG200", "Coffee", "Plane", "Trace", "PowerCons"]
DATASETS = ["SyntheticControl", "Chinatown"]
ENCODERS = [CNN_DFS, ResNet_DFS, RNN_TS, CNN_TS, ResNet_TS]
ENCODERS = [CNN_DFS]
# ~~~~~~~~~~~~~~~~~~~~~~~
RHO_DFS: float = 0.1
BATCH_SIZE: bool = 128
WINDOW_LENGTH: int = 5
WINDOW_STRIDE: int = 1
# ~~~~~~~~~~~~~~~~~~~~~~~
QUANT_INTERVALS: int = 5 
QUANT_SHIFTS: list[float] = [0.0] 
# ~~~~~~~~~~~~~~~~~~~~~~~
PRE_MAXEPOCH: int = 60
TRA_MAXEPOCH: int = 120
LEARNING_RATE: float = 1E-4
# ~~~~~~~~~~~~~~~~~~~~~~~
DIR_CACHE = Path("cache/")
DIR_TRAIN = Path("training/")
DIR_RESULTS = Path("results/")
# ~~~~~~~~~~~~~~~~~~~~~~~
NSPLITS = 5
RANDOM_STATE = 0

# =================================

exp_dict = {"base": EXP_base, "ratio": EXP_ratio, "quant": EXP_quant, "stride": EXP_stride}
EXP = exp_dict[EXP]

for i, (arch, dataset) in enumerate(product(ENCODERS, DATASETS)):

    log.info(f"Current dataset: {dataset}")
    log.info(f"Current decoder: {arch.__str__()}")
    X, Y, mapping = download_dataset(dataset_name=dataset, dir_cache=DIR_CACHE)

    log.info(f"Train-test K-Fold validation: ({NSPLITS} splits)")
    skf = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=RANDOM_STATE)
    for j, (train_index, test_index) in enumerate(skf.split(X, Y)):

        X_train, Y_train = X[train_index,:], Y[train_index]
        X_test, Y_test = X[test_index,:], Y[test_index]

        EXP(dataset=dataset, arch=arch, 
            X_train=X_train, Y_train=Y_train, 
            X_test=X_test, Y_test=Y_test,
            # ~~~~~~~~~~~~~~~~~~~~~~~
            rho_dfs=RHO_DFS,
            batch_size=BATCH_SIZE,
            window_length=WINDOW_LENGTH,
            window_stride=WINDOW_STRIDE,
            quant_intervals=QUANT_INTERVALS,
            quant_shifts=QUANT_SHIFTS,
            # ~~~~~~~~~~~~~~~~~~~~~~~
            dir_cache=DIR_CACHE,
            dir_train=DIR_TRAIN,
            dir_results=DIR_RESULTS,
            # ~~~~~~~~~~~~~~~~~~~~~~~
            pre_maxepoch=PRE_MAXEPOCH, 
            tra_maxepoch=TRA_MAXEPOCH,
            learning_rate=LEARNING_RATE,
            # ~~~~~~~~~~~~~~~~~~~~~~~
            fold_number=j, total_folds=NSPLITS,
            random_state=RANDOM_STATE)
        