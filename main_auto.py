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
from s3ts.experiments import EXP_ratio

from itertools import product
import logging

import torch

# set up logging
from s3ts import LOGH_FILE, LOGH_CLI
log = logging.getLogger(__name__)
log.addHandler(LOGH_FILE), log.addHandler(LOGH_CLI) 

# SETTINGS
# =================================

NSPLITS = 5
RANDOM_STATE = 0

DATASETS = ["GunPoint", "Coffee", "PowerCons", "Plane", "CBF"]
ENCODERS = [CNN_DFS, ResNet_DFS]

# =================================

for i, (arch, dataset) in enumerate(product(ENCODERS, DATASETS)):

    log.info(f"Current dataset: {dataset}")
    log.info(f"Current decoder: {arch.__str__()}")
    X, Y, mapping = download_dataset(dataset_name=dataset)

    log.info(f"Train-test K-Fold validation: ({NSPLITS} splits)")
    skf = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=RANDOM_STATE)
    for j, (train_index, test_index) in enumerate(skf.split(X, Y)):

        X_train, Y_train = X[train_index,:], Y[train_index]
        X_test, Y_test = X[test_index,:], Y[test_index]

        data = EXP_ratio(dataset=dataset, arch=arch, 
            X_train=X_train, Y_train=Y_train, 
            X_test=X_test, Y_test=Y_test,
            fold_number=j, total_folds=NSPLITS, 
            random_state=RANDOM_STATE)
