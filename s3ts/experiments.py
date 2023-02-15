"""
Functions to perform the experiments presented in the article.
"""

from pytorch_lightning import LightningModule

from s3ts.setup import train_model

from datetime import datetime
import pandas as pd
import numpy as np

from pathlib import Path

# default values
# ~~~~~~~~~~~~~~~~~~~~~~ #

window_size: int = 5
batch_size: int = 128
rho_dfs: float = 0.1

nframes_tra: int = None
nframes_pre: int = None
nframes_test: int = None

# ptask 1 settings: quantile prediction
quant: bool = True
quant_intervals: int = 5
quant_shifts: list[int] = [0]

# TODO ptask 2 settings: frame ordering
order: bool = True
order_nframes: int = 4

# training procedure settings
stop_metric: str = "val_f1"
pre_patience: int = 5
pre_maxepoch: int = 100
tra_patience: int = 40
tra_maxepoch: int = 200

# folders 
dir_cache: Path = Path("cache/")
dir_train: Path = Path("training/exp/")
dir_results: Path = Path("results/")

# ~~~~~~~~~~~~~~~~~~~~~~ #

# =====================================================
# =====================================================
# AUXILIARY FUNCTIONS
# =====================================================
# =====================================================

def create_folders() -> None:
    """ Ensures all needed folders exist."""
    for path in [dir_cache, dir_train, dir_results]:
        path.mkdir(parents=True, exist_ok=True)

def base_results(dataset: str, fold_number: int, 
        arch: type[LightningModule], random_state: int = 0) -> pd.DataFrame:
    
    df = pd.Series(dtype="object")
    df["dataset"], df["arch"]  = dataset, arch.__str__()
    df["fold_number"], df["random_state"] = fold_number, random_state
    df["batch_size"], df["window_size"] = batch_size, window_size
    
    df["nframes_tra"], df["nframes_pre"], df["nframes_test"] = nframes_tra, nframes_pre, nframes_test

    pass

# =====================================================
# =====================================================
# EXPERIMENTS
# =====================================================
# =====================================================

def EXP_ratio(
    dataset: str, arch: type[LightningModule],
    X_train: np.ndarray,  X_test: np.ndarray, 
    Y_train: np.ndarray,  Y_test: np.ndarray,
    fold_number: int = 0, random_state: int = 0,
    ) -> pd.DataFrame:

    """ Experiment to check the effect of train/pretrain sample ratios."""

    # make sure folders exist
    create_folders() 
    
    # prepare the data
    train_dm, pretrain_dm = prepare_data_modules(dataset=dataset,
        X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
        batch_size=batch_size, window_size=window_size, lab_shifts=lab_shifts,
        rho_dfs=rho_dfs, pret_frac=pret_frac, fold_number=fold_number,
        nframes_tra=nframes_tra, nframes_pre=nframes_pre, nframes_test=nframes_test,
        seed_sts=seed_sts, seed_label=seed_label, pre_intervals=pre_intervals)

    # run the base model
    model, checkpoint, data_aux = train_model(
            directory=subdir_train, label="def", 
            epoch_max=pre_maxepoch, epoch_patience=pre_patience)

    # run with different ratios
    runs = list()
    RATIOS = [1, 2, 4, 8]
    for i, ratio in enumerate(RATIOS):

        date_flag = datetime.now().strftime("%Y-%m-%d_%H-%M")
        subdir_train = dir_train / f"EXP_ratio_f{fold_number}.{i}_{date_flag}"

        # TODO write sth to manage this
        train_dm.set_percent_whatever()

        model, checkpoint, data_aux = train_model(
            directory=subdir_train, label="aux", 
            epoch_max=pre_maxepoch, epoch_patience=pre_patience)


        model, checkpoint, data_aux = train_model(
            directory=subdir_train, label="pre", 
            epoch_max=pre_maxepoch, epoch_patience=pre_patience)

        runs.append(data)
        runs_df = pd.concat(runs, ignore_index=True)
        runs_df.to_csv("results/results.csv", index=False)
    
    return runs_df

# =====================================================

def EXP_frames(
    dataset: str, arch: type[LightningModule],
    X_train: np.ndarray,  X_test: np.ndarray, 
    Y_train: np.ndarray,  Y_test: np.ndarray,
    fold_number: int = 0, random_state: int = 0,
    ):

    nframes_tra: int = None 
    nframes_pre: int = None
    nframes_test: int = None

    pass

# =====================================================

def EXP_quantiles(
    dataset: str, directory: Path, arch: type[LightningModule],
    X_train: np.ndarray,  X_test: np.ndarray, 
    Y_train: np.ndarray,  Y_test: np.ndarray,
    fold_number: int = 0, random_state: int = 0,
    ):

    INTERVALS = [3,5,7]

    pass

def EXP_shifts(
    dataset: str, directory: Path, arch: type[LightningModule],
    X_train: np.ndarray,  X_test: np.ndarray, 
    Y_train: np.ndarray,  Y_test: np.ndarray,
    fold_number: int = 0, random_state: int = 0,
    ):

    SHIFTS = [3,5,7]

    pass

def EXP_sorting(
    dataset: str, directory: Path, arch: type[LightningModule],
    X_train: np.ndarray,  X_test: np.ndarray, 
    Y_train: np.ndarray,  Y_test: np.ndarray,
    fold_number: int = 0, random_state: int = 0,
    ):

    pass

def EXP_comparison():

    pass


# =====================================================

def compare_pretrain(
    dataset: str,
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    Y_train: np.ndarray, 
    Y_test: np.ndarray,
    directory: Path,
    arch: type[LightningModule],
    # ~~~~~~~~~~~~~~~~
    batch_size: int,
    window_size: int,
    pre_intervals: int,
    lab_shifts: list[int],
    # ~~~~~~~~~~~~~~~~
    # they are needed for frame creation and imply recalcs
    rho_dfs: float,
    pret_frac: float,
    # ~~~~~~~~~~~~~~~~
    nframes_tra: int, 
    nframes_pre: int,
    nframes_test: int,
    # ~~~~~~~~~~~~~~~~
    pre_patience: int = 5,
    pre_maxepoch: int = 100,
    tra_patience: int = 40,
    tra_maxepoch: int = 200,
    # ~~~~~~~~~~~~~~~~
    stop_metric: str = "val_f1",
    # ~~~~~~~~~~~~~~~~
    seed_sts: int = 0,
    seed_label: int = 0,
    seed_torch: int = 0,
    fold_number: int = 0,
    ) -> pd.Series:

    seed_everything(seed_torch)

    results = pd.Series(dtype="object")
    results["dataset"], results["fold_number"], results["decoder"] = dataset, fold_number, arch.__str__()
    results["seed_sts"], results["seed_label"], results["fold_number"] = seed_sts, seed_label, fold_number
    results["batch_size"], results["window_size"], results["lab_shifts"] = batch_size, window_size, lab_shifts
    results["nframes_tra"], results["nframes_pre"],results["nframes_test"] = nframes_tra, nframes_pre, nframes_test
    
    train_dm, pretrain_dm = prepare_data_modules(dataset=dataset,
        X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
        batch_size=batch_size, window_size=window_size, lab_shifts=lab_shifts,
        rho_dfs=rho_dfs, pret_frac=pret_frac, fold_number=fold_number,
        nframes_tra=nframes_tra, nframes_pre=nframes_pre, nframes_test=nframes_test,
        seed_sts=seed_sts, seed_label=seed_label, pre_intervals=pre_intervals)

    train_dm: FramesDataModule
    pretrain_dm: FramesDataModule

    # ~~~~~~~~~~~~~~~~~~~~~ train without pretrain

    # create the model
    train_model = PredModel(
            n_labels=train_dm.n_labels, 
            n_patterns=train_dm.n_patterns,
            l_patterns=train_dm.l_patterns,
            window_size=train_dm.window_size,
            lab_shifts=[0],
            arch=arch) 

    # train the model
    trainer, checkpoint = setup_trainer(directory=directory,  version="def",
        epoch_max=tra_maxepoch, epoch_patience=tra_patience, stop_metric=stop_metric)
    trainer.fit(train_model, datamodule=train_dm)
    train_model = train_model.load_from_checkpoint(checkpoint.best_model_path)

    valid_results = trainer.validate(train_model, datamodule=train_dm)
    test_results = trainer.test(train_model, datamodule=train_dm)
    
    # log results
    results["def_val_acc"] = valid_results[0]["val_acc"]
    results["def_val_f1"] = valid_results[0]["val_f1"]
    results["def_val_auroc"] = valid_results[0]["val_auroc"]

    results["def_test_acc"] = test_results[0]["test_acc"]
    results["def_test_f1"] = test_results[0]["test_f1"]
    results["def_test_auroc"] = test_results[0]["test_auroc"]

    results["def_best_model"] = checkpoint.best_model_path
    results["def_train_csv"] = str(directory  / "logs" / "def" / "metrics.csv")
    results["def_nepochs"] = pd.read_csv(results["def_train_csv"])["epoch_train_acc"].count()

    # ~~~~~~~~~~~~~~~~~~~~~ do the pretrain

    # create the model
    pretrain_model = PredModel(
            n_labels=pretrain_dm.n_labels, 
            n_patterns=pretrain_dm.n_patterns,
            l_patterns=pretrain_dm.l_patterns,
            window_size=pretrain_dm.window_size,
            lab_shifts=pretrain_dm.lab_shifts,
            arch=arch)
               
    trainer, checkpoint = setup_trainer(directory=directory,  version="aux",
        epoch_max=pre_maxepoch, epoch_patience=pre_patience, stop_metric=stop_metric)
    trainer.fit(train_model, datamodule=train_dm)
    pretrain_model = pretrain_model.load_from_checkpoint(checkpoint.best_model_path)
    valid_results = trainer.validate(train_model, datamodule=train_dm)

    # log results
    results["aux_val_acc"] = valid_results[0]["val_acc"]
    results["aux_val_f1"] = valid_results[0]["val_f1"]

    results["aux_best_model"] = checkpoint.best_model_path
    results["aux_train_csv"] = str(directory  / "logs" / "aux" / "metrics.csv")
    results["aux_nepochs"] = pd.read_csv(results["aux_train_csv"])["epoch_train_acc"].count()

    # grab the pretrained encoder
    pretrained_encoder = pretrain_model.encoder

    # ~~~~~~~~~~~~~~~~~~~~~ train with pretrain

    train_model = PredModel(
            n_labels=train_dm.n_labels, 
            n_patterns=train_dm.n_patterns,
            l_patterns=train_dm.l_patterns,
            window_size=train_dm.window_size,
            lab_shifts=[0],
            arch=arch) 
    train_model.encoder = pretrained_encoder
    trainer, checkpoint = setup_trainer(directory=directory,  version="pre",
        epoch_max=tra_maxepoch, epoch_patience=tra_patience, stop_metric=stop_metric)
    trainer.fit(train_model, datamodule=train_dm)
    train_model = train_model.load_from_checkpoint(checkpoint.best_model_path)
    valid_results = trainer.validate(train_model, datamodule=train_dm)
    test_results = trainer.test(train_model, datamodule=train_dm)

    # log results
    results["pre_val_acc"] = valid_results[0]["val_acc"]
    results["pre_val_f1"] = valid_results[0]["val_f1"]
    results["pre_val_auroc"] = valid_results[0]["val_auroc"]

    results["pre_test_acc"] = test_results[0]["test_acc"]
    results["pre_test_f1"] = test_results[0]["test_f1"]
    results["pre_test_auroc"] = test_results[0]["test_auroc"]

    results["pre_best_model"] = checkpoint.best_model_path
    results["pre_train_csv"] = str(directory  / "logs" / "pre" / "metrics.csv")
    results["pre_nepochs"] = pd.read_csv(results["pre_train_csv"])["epoch_train_acc"].count()
    
    print("\nTraining summary:")
    print(results)

    return results.to_frame().transpose().copy()