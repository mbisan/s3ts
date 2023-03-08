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
from s3ts.experiments import EXP_ratio, EXP_quantiles

from itertools import product
import argparse
import logging

import torch
torch.set_float32_matmul_precision("medium")

# set up logging
from s3ts import LOGH_FILE, LOGH_CLI
log = logging.getLogger(__name__)
log.addHandler(LOGH_FILE), log.addHandler(LOGH_CLI) 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='''Perform the experiments from the article [title] by R. Coterillo and A. Pérez.''')

    parser.add_argument('--dataset', type=str, required=True,
                        choices = ["GunPoint", "Coffee", "PowerCons", "Plane", "CBF"],
                        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--mode', type=str, required=True, choices=['DF', 'TS'],
                        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--architecture', type=str, required=True, choices=['RNN', 'CNN', 'ResNet'],
                        help='Name of the architecture from which create the model')
    
    parser.add_argument('--experiment', type=str, required=True, choices=['ratio', 'quantiles'],
                        help='Name of the architecture from which create the model')

    parser.add_argument('--window_size', type=int, default=5,
                        help='Window size used for training')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used during training')
    
    parser.add_argument('--rho_df', type=float, default=0.1,
                        help='Value of the forgetting parameter')

    parser.add_argument('--lr', type=float, default=5e-03,
                        help='Value of the learning rate')

    parser.add_argument('--max_epochs', type=int, default=15,
                        help='Number of epochs to train the network')
    
    parser.add_argument('--num_splits', type=int, default=5,
                        help='Number of splits for K-fold validation')

    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of epochs to train the network')
    
    parser.add_argument('--random_state', type=int, default=0,
                        help='Global seed for the random number generators')

    parser.add_argument('--PATH', type=str, default=None,
                        help='path to the checkpoint, if provided it will no train the network but only evaluate it')

    args = parser.parse_args()

    # get model class
    arch_dict = {"TS": {"RNN": RNN_TS, "CNN": CNN_TS, "ResNet": ResNet_TS}, "DF": {"CNN": CNN_DFS, "ResNet": ResNet_DFS}}
    if args.architecture not in arch_dict[args.mode]:
        raise NotImplementedError("Invalid mode and architecture combination!")
    arch = arch_dict[args.mode][args.architecture]

    # get experiment function
    exp_dict = {"ratio": EXP_ratio, "quantiles": EXP_quantiles}
    exp = exp_dict[args.experiment]

    # download dataset
    dataset: str = args.dataset
    X, Y, mapping = download_dataset(dataset_name=dataset)

    n_splits: int = args.num_splits
    log.info(f"Train-test K-Fold validation: ({n_splits} splits)")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
    for j, (train_index, test_index) in enumerate(skf.split(X, Y)):

        X_train, Y_train = X[train_index,:], Y[train_index]
        X_test, Y_test = X[test_index,:], Y[test_index]

        exp(dataset=dataset, arch=arch, 
            X_train=X_train, Y_train=Y_train, 
            X_test=X_test, Y_test=Y_test,
            fold_number=j, total_folds=n_splits, 
            random_state=args.random_state)
