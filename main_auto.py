"""
Automatic experimentation script.

@author Raúl Coterillo
@version 2023-02
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
from s3ts.experiments import EXP_ratio, EXP_quant

from itertools import product
from pathlib import Path
import logging

import torch
torch.set_float32_matmul_precision("medium")

# set up logging
from s3ts import LOGH_FILE, LOGH_CLI
log = logging.getLogger(__name__)
log.addHandler(LOGH_FILE), log.addHandler(LOGH_CLI) 

# SETTINGS
# =================================

EXP = "ratio"
DATASETS = ["GunPoint", "Coffee", "PowerCons", "Plane", "CBF"]
ENCODERS = [CNN_DFS, ResNet_DFS, RNN_TS, CNN_TS, ResNet_TS]
# ~~~~~~~~~~~~~~~~~~~~~~~
RHO_DFS: float = 0.1
BATCH_SIZE: bool = 128
WINDOW_SIZE: int = 5
# ~~~~~~~~~~~~~~~~~~~~~~~
QUANT_INTERVALS: int = 5 
QUANT_SHIFTS: list[float] = [0.0] 
# ~~~~~~~~~~~~~~~~~~~~~~~
PRE_MAXEPOCH: int = 60
TRA_MAXEPOCH: int = 120
LEARNING_RATE: float = 1E-4
# ~~~~~~~~~~~~~~~~~~~~~~~
DIR_CACHE = Path("cache/")
DIR_TRAIN = Path("training/exp")
DIR_RESULTS = Path("results/")
# ~~~~~~~~~~~~~~~~~~~~~~~
NSPLITS = 5
RANDOM_STATE = 0

# =================================

exp_dict = {"ratio": EXP_ratio, "quant": EXP_quant}
for i, (arch, dataset) in enumerate(product(ENCODERS, DATASETS)):

    log.info(f"Current dataset: {dataset}")
    log.info(f"Current decoder: {arch.__str__()}")
    X, Y, mapping = download_dataset(dataset_name=dataset)

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
            window_size=WINDOW_SIZE,
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
        