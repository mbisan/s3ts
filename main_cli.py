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

# standard library
from pathlib import Path
import logging as log
import argparse

# torch configuration
import torch
torch.set_float32_matmul_precision("medium")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='''Perform the experiments showcased in the article.''')

    parser.add_argument('--dataset', type=str, required=True,
                        choices = ["GunPoint", "Coffee", "PowerCons", "Plane", "CBF"],
                        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--mode', type=str, required=True, choices=['DF', 'TS'],
                        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--arch', type=str, required=True, choices=['RNN', 'CNN', 'ResNet'],
                        help='Name of the architecture from which create the model')
    
    parser.add_argument('--exp', type=str, required=True, choices=['ratio', 'quant'],
                        help='Name of the architecture from which create the model')

    parser.add_argument('--rho_dfs', type=float, default=0.1,
                        help='Value of the forgetting parameter')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used during training')

    parser.add_argument('--window_size', type=int, default=5,
                        help='Window size used for training')
    
    parser.add_argument('--quant_intervals', type=int, default=5,
                        help='Number of quantiles used in pretraining')
    
    parser.add_argument('--quant_shift', type=float, default=0.0,
                        help='Label shift during pretraining')

    parser.add_argument('--pre_maxepoch', type=int, default=60,
                        help='Number of epochs to pretrain the networks')
    
    parser.add_argument('--tra_maxepoch', type=int, default=120,
                        help='Number of epochs to train the networks')
    
    parser.add_argument('--learning_rate', type=float, default=1e-04,
                        help='Value of the learning rate')
    
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of splits for K-fold validation')
    
    parser.add_argument('--random_state', type=int, default=0,
                        help='Global seed for the random number generators')

    parser.add_argument('--dir_cache', type=str, default="cache/",
                        help='Directory for the cached data')
    
    parser.add_argument('--dir_train', type=str, default="training/exp",
                        help='Directory for the training files')
    
    parser.add_argument('--dir_results', type=str, default="results/",
                        help='Directory for the results (CSVs)')
    
    parser.add_argument('--log_file', type=str, default="debug.log",
                        help='Directory for the results (CSVs)')

    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # sanity checks and variable selectors
    arch_dict = {"TS": {"RNN": RNN_TS, "CNN": CNN_TS, "ResNet": ResNet_TS}, "DF": {"CNN": CNN_DFS, "ResNet": ResNet_DFS}}
    if args.arch not in arch_dict[args.mode]:
        raise NotImplementedError("Invalid mode and architecture combination!")
    exp_dict = {"ratio": EXP_ratio, "quant": EXP_quant}
    
    # get rest of variables
    dataset: str = args.dataset
    arch = arch_dict[args.mode][args.arch]
    exp = exp_dict[args.exp]
    # ~~~~~~~~~~~~~~~~~~~~~~~
    rho_dfs: float = args.rho_dfs
    batch_size: int = args.batch_size
    window_size: int = args.window_size
    quant_shifts: list[float] = [args.quant_shift]
    quant_intervals: int = args.quant_intervals
    # ~~~~~~~~~~~~~~~~~~~~~~~
    pre_maxepoch: int = args.pre_maxepoch
    tra_maxepoch: int = args.tra_maxepoch
    learning_rate: float = args.learning_rate
    # ~~~~~~~~~~~~~~~~~~~~~~~
    dir_cache: Path = Path(args.dir_cache)
    dir_train: Path = Path(args.dir_train)
    dir_results: Path = Path(args.dir_results)
    log_file: Path = Path(args.log_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~
    n_splits: int = args.n_splits
    random_state: int = args.random_state
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # logging setup
    log.basicConfig(filename=log_file, level=log.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')

    # load dataset
    dataset: str = args.dataset
    X, Y, mapping = download_dataset(dataset_name=dataset, dir_cache=dir_cache)
    
    log.info(f"Train-test K-Fold validation: ({n_splits} splits)")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
    for j, (train_index, test_index) in enumerate(skf.split(X, Y)):

        X_train, Y_train = X[train_index,:], Y[train_index]
        X_test, Y_test = X[test_index,:], Y[test_index]
        
        exp(dataset=dataset, arch=arch, 
            X_train=X_train, Y_train=Y_train, 
            X_test=X_test, Y_test=Y_test,
            # ~~~~~~~~~~~~~~~~~~~~~~~
            rho_dfs=rho_dfs,
            batch_size=batch_size,
            window_size=window_size,
            quant_intervals=args.quant_intervals,
            quant_shifts=quant_shifts,
            # ~~~~~~~~~~~~~~~~~~~~~~~
            dir_cache=dir_cache,
            dir_train=dir_train,
            dir_results=dir_results,
            # ~~~~~~~~~~~~~~~~~~~~~~~
            pre_maxepoch=pre_maxepoch, 
            tra_maxepoch=tra_maxepoch,
            learning_rate=learning_rate,
            # ~~~~~~~~~~~~~~~~~~~~~~~
            fold_number=j, total_folds=n_splits,
            random_state=random_state)
