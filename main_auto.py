"""
Automatic training under several conditions.

@author Raúl Coterillo
@version 2023-01
"""

# data
from s3ts.frames.tasks.download import download_dataset
from sklearn.model_selection import StratifiedKFold

# architectures
from s3ts.models.encoders.ResNet import ResNet_Encoder
from s3ts.models.encoders.CNN import CNN_Encoder

# experiments
from s3ts.experiments import EXP_ratio

from itertools import product
import logging

import torch
torch.set_float32_matmul_precision("medium")

# set up logging
from s3ts import LOGH_FILE, LOGH_CLI
log = logging.getLogger(__name__)
log.addHandler(LOGH_FILE), log.addHandler(LOGH_CLI) 

# SETTINGS
# =================================

NSPLITS = 5
RANDOM_STATE = 0

DATASETS = ["GunPoint"]
ENCODERS = [CNN_Encoder]#, ResNet_Encoder]

# BUG WITH THE SPLIT, MORE TRAIN SAMPLES THAN SHOULD

# =================================

for i, (dataset, arch) in enumerate(product(DATASETS, ENCODERS)):

    log.info(f"Current dataset: {dataset}")
    log.info(f"Current dataset: {arch.__str__()}")
    X, Y, mapping = download_dataset(dataset_name=dataset)

    log.info(f"Train-test K-Fold validation: ({NSPLITS} splits)")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    for j, (train_index, test_index) in enumerate(skf.split(X, Y)):

        log.info(f"Fold {j}:")
        X_train, Y_train = X[train_index,:], Y[train_index]
        X_test, Y_test = X[test_index,:], Y[test_index]

        data = EXP_ratio(dataset=dataset, arch=arch, 
            X_train=X_train, Y_train=Y_train, 
            X_test=X_test, Y_test=Y_test,
            fold_number=j, total_folds=NSPLITS, 
            random_state=RANDOM_STATE)
