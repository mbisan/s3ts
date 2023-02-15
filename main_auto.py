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

# import torch
# torch.set_float32_matmul_precision("medium")

# SETTINGS
# =================================

# add a progress logger

NSPLITS = 10
RANDOM_STATE = 0

DATASETS = ["GunPoint"]
ENCODERS = [CNN_Encoder, ResNet_Encoder]

# =================================

for i, (dataset, arch) in enumerate(product(DATASETS, ENCODERS)):

    X, Y, mapping = download_dataset(dataset_name=dataset)

    print(f"Train-test K-Fold validation: ({NSPLITS} splits)")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    for j, (train_index, test_index) in enumerate(skf.split(X, Y)):

        print(f"Fold {j}:")
        X_train, Y_train = X[train_index,:], Y[train_index]
        X_test, Y_test = X[test_index,:], Y[test_index]

        data = EXP_ratio(dataset=dataset, arch=arch, 
            X_train=X_train, Y_train=Y_train, 
            X_test=X_test, Y_test=Y_test,
            fold_number=j, random_state=RANDOM_STATE)
