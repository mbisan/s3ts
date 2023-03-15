#/usr/bin/python3

"""
Functions to perform the experiments presented in the article.
"""

# data processing stuff
from s3ts.data.tasks.compute import compute_medoids, compute_STS
from s3ts.data.tasks.oesm import compute_OESM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# models / modules
from pytorch_lightning import LightningModule
from s3ts.data.modules import DoubleDataModule
from s3ts.models.wrapper import WrapperModel

# training stuff
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning import Trainer

from datetime import datetime
from pathlib import Path
import logging as log
import pandas as pd
import numpy as np

# =====================================================
# =====================================================
# AUXILIARY FUNCTIONS
# =====================================================
# =====================================================

def create_folders(
        dir_cache: Path = Path("cache/"),
        dir_train: Path = Path("training/exp/"),
        dir_results: Path = Path("results/")
        ) -> None:

    """ Ensures all needed folders exist."""
    log.info("Creating folders...")
    for path in [dir_cache, dir_train, dir_results]:
        path.mkdir(parents=True, exist_ok=True)
        log.info("..." + str(path))
    log.info("Done!")

def prepare_dms(
        dataset: str, 
        X_train: np.ndarray, X_test: np.ndarray, 
        Y_train: np.ndarray, Y_test: np.ndarray,
        rho_dfs: float, pret_frac: float,
        # ~~ # NOTE: can be changed without recalcs # ~~ #
        batch_size: int, window_length: int, window_stride: int,                  
        quant_shifts: list[int], quant_intervals: int,     
        # multipliers for the number of frames generated
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        nsamp_tra: float = None, nsamp_pre: float = None, nsamp_test: float = None,
        # cross validation stuff
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fold_number: int = 0, random_state: int = 0, frames: bool = True,
        dir_cache: Path = Path("cache/"),
        ) -> tuple[DoubleDataModule, DoubleDataModule]:

    """ Prepares the data modules for the training. """

    # print dataset info
    log.info("~~~~~~~~~~~~~~~~~~~~~~~~~")
    log.info(f"         Dataset: {dataset}")
    log.info(f"     Fold number: {fold_number}")
    log.info(f"   Total samples: {X_train.shape[0] + X_test.shape[0]}")
    
    # prtrain/train set splitting
    log.debug(f"Splitting train and pretrain sets (seed: {random_state})")
    X_tra, X_pre, Y_tra, Y_pre = train_test_split(X_train, Y_train, 
        test_size=pret_frac, stratify=Y_train, random_state=random_state, shuffle=True)

    # print more dataset info
    log.info(f"Pretrain samples: {X_pre.shape[0]}")
    log.info(f"   Train samples: {X_tra.shape[0]}")
    log.info(f"    Test samples: {X_test.shape[0]}")
    log.info("~~~~~~~~~~~~~~~~~~~~~~~~~")

    # pattern selection: shape = [n_patterns,  l_patterns]
    log.info(f"Selecting the DFS patterns from the train data")
    medoids, medoid_ids = compute_medoids(X_tra, Y_tra, distance_type="dtw")

    log.info("Generating 'train' STS...")       # train STS generation
    STS_tra, labels_tra, frames_tra = compute_STS(X=X_tra,Y=Y_tra, target_nframes=nsamp_tra, 
        frame_buffer=window_length*3, random_state=random_state)

    log.info("Generating 'pretrain' STS...")    # pretrain STS generation
    STS_pre, _, frames_pre = compute_STS(X=X_pre, Y=Y_pre, target_nframes=nsamp_pre, 
        frame_buffer=window_length*3, random_state=random_state)
    
    kbd = KBinsDiscretizer(n_bins=quant_intervals, encode="ordinal", strategy="quantile", random_state=random_state)
    kbd.fit(STS_pre.reshape(-1,1))
    labels_pre = kbd.transform(STS_pre.reshape(-1,1)).squeeze().astype(int)
    
    log.info("Generating 'test' STS...")        # test STS generation
    STS_test, labels_test, frames_test = compute_STS(X=X_test, Y=Y_test, target_nframes=nsamp_test, 
        frame_buffer=window_length*3,random_state=random_state)

    # DFS generation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fracs_str = f"pf{pret_frac}"
    seeds_str = f"f{fold_number}-rs{random_state}"
    frames_str = f"tr{frames_tra}-pr{frames_pre}-ts{frames_test}"
    cache_file = dir_cache / f"DFS_{dataset}_{fracs_str}_{seeds_str}_{frames_str}.npz"
    if not Path(cache_file).exists():
        log.info("Generating 'train' DFS...")
        DFS_tra = compute_OESM(STS_tra, medoids, rho=rho_dfs)
        log.info("Generating 'pretrain' DFS...")
        DFS_pre = compute_OESM(STS_pre, medoids, rho=rho_dfs) 
        log.info("Generating 'test' DFS...")
        DFS_test = compute_OESM(STS_test, medoids, rho=rho_dfs) 
        np.savez_compressed(cache_file, DFS_tra=DFS_tra, DFS_pre=DFS_pre, DFS_test=DFS_test)
    else:
        log.info(f"Loading DFS from cached file... ({cache_file})")
        with np.load(cache_file) as data:
            DFS_tra, DFS_pre, DFS_test = data["DFS_tra"], data["DFS_pre"], data["DFS_test"]

    log.info("Creating 'train' dataset...")
    dm_tra = DoubleDataModule(
        STS_train=STS_tra, DFS_train=DFS_tra, labels_train=labels_tra, nsamp_train=frames_tra,
        STS_test=STS_test, DFS_test=DFS_test, labels_test=labels_test, nsamp_test=frames_test,
        window_length=window_length, window_stride=window_stride, batch_size=batch_size, 
        quant_shifts=[0], frames=frames, patterns=medoids)

    log.info("Creating 'pretrain' dataset...")
    quant_shifts = np.round(np.array(quant_shifts)*X_train.shape[1]).astype(int)
    log.info(f"Number of quantiles: {quant_intervals}")
    log.info(f"Label shifts: {quant_shifts}")    

    # create data module (pretrain)
    dm_pre = DoubleDataModule(
        STS_train=STS_pre, DFS_train=DFS_pre, labels_train=labels_pre, nsamp_train=frames_pre,
        window_length=window_length, window_stride=window_stride, batch_size=batch_size, 
        quant_shifts=quant_shifts, frames=frames, patterns=medoids)   

    return dm_tra, dm_pre

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def setup_trainer(
    directory: Path,
    version: str,
    epoch_max: int,
    stop_metric: str = "val_acc",
    ) -> tuple[Trainer, ModelCheckpoint]:

    """ Shared setup for the Trainer objects. """

    checkpoint = ModelCheckpoint(monitor=stop_metric, mode="max")    
    trainer = Trainer(default_root_dir=directory,  accelerator="auto",
        # progress logs
        logger = [
            TensorBoardLogger(save_dir=directory, name="logs", version=version),
            CSVLogger(save_dir=directory, name="logs", version=version)
        ],
        callbacks=[
            # early stop the model
            # EarlyStopping(monitor=stop_metric, mode="max", patience=epoch_patience),         
            LearningRateMonitor(logging_interval='step'),  # learning rate logger
            checkpoint  # save best model version
            ],
        max_epochs=epoch_max,  deterministic = False,
        log_every_n_steps=1, check_val_every_n_epoch=1
    )

    return trainer, checkpoint

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_model(
    directory: Path,
    label: str,
    epoch_max: int,
    dm: DoubleDataModule,
    arch: type[LightningModule],
    stop_metric: str = "val_acc",
    encoder: LightningModule = None,
    learning_rate: float = 1e-4,
    ) -> tuple[pd.DataFrame, WrapperModel, ModelCheckpoint]:

    results = pd.Series(dtype="object")
    frames: bool = arch.__frames__()

    # create the model
    if frames:
        model = WrapperModel(
                n_labels=dm.n_labels, 
                n_patterns=dm.n_patterns,
                l_patterns=dm.l_patterns,
                window_length=dm.window_length,
                lab_shifts=[0],
                arch=arch, 
                learning_rate=learning_rate)
    else:
        model = WrapperModel(
                n_labels=dm.n_labels, 
                n_patterns=1,
                l_patterns=1,
                window_length=dm.window_length,
                lab_shifts=[0],
                arch=arch, 
                learning_rate=learning_rate)
    
    # set encoder if one was passed
    if encoder is not None:
        model.encoder = encoder

    # train the model
    trainer, checkpoint = setup_trainer(directory=directory,  version=label,
        epoch_max=epoch_max, stop_metric=stop_metric)
    trainer.fit(model, datamodule=dm)

    # load best checkpoint
    model = model.load_from_checkpoint(checkpoint.best_model_path)

    # log val results
    val_results = trainer.validate(model, datamodule=dm)
    results[f"{label}_val_acc"] = val_results[0]["val_acc"]
    results[f"{label}_val_f1"] = val_results[0]["val_f1"]
    results[f"{label}_val_auroc"] = val_results[0]["val_auroc"]

    # log test results
    if dm.test:
        test_results = trainer.test(model, datamodule=dm)
        results[f"{label}_test_acc"] = test_results[0]["test_acc"]
        results[f"{label}_test_f1"] = test_results[0]["test_f1"]
        results[f"{label}_test_auroc"] = test_results[0]["test_auroc"]

    # load model info
    results[f"{label}_best_model"] = checkpoint.best_model_path
    results[f"{label}_train_csv"] = str(directory  / "logs" / label / "metrics.csv")
    results[f"{label}_nepochs"] = pd.read_csv(results[f"{label}_train_csv"])["epoch_train_acc"].count()
    results = results.to_frame().transpose().copy()

    return results, model, checkpoint

def base_results(dataset: str, fold_number: int, 
        arch: type[LightningModule], pretrained: bool, 
        batch_size: int, window_length: int, window_stride: int,
        random_state: int = 0) -> pd.DataFrame:
    
    """ Series template for the results. """

    df = pd.Series(dtype="object")
    df["dataset"], df["arch"], df["pretrained"]  = dataset, arch.__str__(), pretrained
    df["fold_number"], df["random_state"] = fold_number, random_state
    df["batch_size"], df["window_length"], df["window_stride"] = batch_size, window_length, window_stride
    df = df.to_frame().transpose().copy()

    return df