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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning import Trainer, seed_everything

from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# set up logging
from s3ts import LOGH_FILE, LOGH_CLI
log = logging.getLogger(__name__)
log.addHandler(LOGH_FILE), log.addHandler(LOGH_CLI) 

# default values
# ~~~~~~~~~~~~~~~~~~~~~~ #

window_size: int = 5
batch_size: int = 128
rho_dfs: float = 0.1

nsamp_tra: int = None
nsamp_pre: int = None
nsamp_test: int = None

# ptask settings: quantile prediction
quant: bool = True
quant_intervals: int = 5
quant_shifts: list[int] = [0]

# training procedure settings
stop_metric: str = "val_acc"
pre_maxepoch: int = 60
tra_maxepoch: int = 240

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
        batch_size: int, window_size: int,                  # NOTE: can be changed without recalcs
        # ptask 1 settings: quantile prediction
        quant_shifts: list[int], quant_intervals: int,      # NOTE: can be changed without recalcs
        # multipliers for the number of frames generated
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        nsamp_tra: float = None, nsamp_pre: float = None, nsamp_test: float = None,
        # cross validation stuff
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fold_number: int = 0, random_state: int = 0, frames: bool = True
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
        frame_buffer=window_size*3,random_state=random_state)

    log.info("Generating 'pretrain' STS...")    # pretrain STS generation
    STS_pre, _, frames_pre = compute_STS(X=X_pre, Y=Y_pre, target_nframes=nsamp_pre, 
        frame_buffer=window_size*3,random_state=random_state)
    
    kbd = KBinsDiscretizer(n_bins=quant_intervals, encode="ordinal", strategy="quantile", random_state=random_state)
    kbd.fit(STS_pre.reshape(-1,1))
    labels_pre = kbd.transform(STS_pre.reshape(-1,1)).squeeze().astype(int)
    
    log.info("Generating 'test' STS...")        # test STS generation
    STS_test, labels_test, frames_test = compute_STS(X=X_test, Y=Y_test, target_nframes=nsamp_test, 
        frame_buffer=window_size*3,random_state=random_state)

    # DFS generation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fracs = f"pf{pret_frac}"
    seeds = f"f{fold_number}-rs{random_state}"
    frames = f"tr{frames_tra}-pr{frames_pre}-ts{frames_test}"
    cache_file = dir_cache / f"DFS_{dataset}_{fracs}_{seeds}_{frames}.npz"
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
        window_size=window_size, batch_size=batch_size, quant_shifts=[0])

    log.info("Creating 'pretrain' dataset...")
    quant_shifts = np.round(np.array(quant_shifts)*X_train.shape[1]).astype(int)
    log.info(f"Number of quantiles: {quant_intervals}")
    log.info(f"Label shifts: {quant_shifts}")    

    # create data module (pretrain)
    dm_pre = DoubleDataModule(
        STS_train=STS_pre, DFS_train=DFS_pre, labels_train=labels_pre, nsamp_train=frames_pre,
        window_size=window_size, batch_size=batch_size, quant_shifts=quant_shifts, frames=frames)   

    return dm_tra, dm_pre

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def setup_trainer(
    directory: Path,
    version: str,
    epoch_max: int,
    stop_metric: str = stop_metric,
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
    stop_metric: str = stop_metric,
    encoder: LightningModule = None,
    ) -> tuple[pd.DataFrame, WrapperModel, ModelCheckpoint]:

    results = pd.Series(dtype="object")

    # create the model
    model = WrapperModel(
            n_labels=dm.n_labels, 
            n_patterns=dm.n_patterns,
            l_patterns=dm.l_patterns,
            window_size=dm.window_size,
            lab_shifts=[0],
            arch=arch)
    
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
        random_state: int = 0) -> pd.DataFrame:
    
    """ Series template for the results. """

    df = pd.Series(dtype="object")
    df["dataset"], df["arch"], df["pretrained"]  = dataset, arch.__str__(), pretrained
    df["fold_number"], df["random_state"] = fold_number, random_state
    df["batch_size"], df["window_size"] = batch_size, window_size
    df = df.to_frame().transpose().copy()

    return df

# =====================================================
# =====================================================
# EXPERIMENTS
# =====================================================
# =====================================================

def EXP_ratio(
    dataset: str, arch: type[LightningModule],
    X_train: np.ndarray,  X_test: np.ndarray, 
    Y_train: np.ndarray,  Y_test: np.ndarray,
    total_folds: int, fold_number: int = 0, random_state: int = 0,
    ) -> pd.DataFrame:

    """ Experiment to check the effect of train/pretrain sample ratios."""

    log.info(f"~~ BEGIN 'ratio' EXPERIMENT (fold #{fold_number+1}/{total_folds}) ~~")

    # make sure folders exist
    create_folders() 
    res_file = dir_results / f"EXP_ratio_{arch.__str__()}_{dataset}_f{fold_number}.csv"

    # NOTE: this is chosen so that the final number of
    # samples for just train and test is the same (50/50 split w/out pretrain)
    pret_frac = 1 - 1/(total_folds-1) 

    # prepare the data
    log.info("Preparing data modules...")
    train_dm, pretrain_dm = prepare_dms(dataset=dataset,
        X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
        batch_size=batch_size, window_size=window_size, 
        rho_dfs=rho_dfs, pret_frac=pret_frac, 
        quant_shifts=quant_shifts, quant_intervals=quant_intervals,
        nsamp_tra=nsamp_tra, nsamp_pre=nsamp_pre, nsamp_test=nsamp_test,
        fold_number=fold_number, random_state=random_state, frames=arch.__frames__())
    train_dm: DoubleDataModule
    pretrain_dm: DoubleDataModule

    runs = []
    PCTS = [0.2, 0.4, 0.6, 0.8, 1]
    trun, crun = len(PCTS)*(1+len(PCTS)), 0
    for i, pct_av_train in enumerate(PCTS):

        # set the train data ratio
        train_dm.ds_train.frac_available = pct_av_train
        train_dm.ds_val.frac_available = pct_av_train
        tot_train_samps = len(train_dm.ds_train) + len(train_dm.ds_val)

        crun += 1
        log.info(f"~ [{crun}/{trun}] Training baseline model... ({pct_av_train*100}% data = {tot_train_samps} samples)")

        # define the training directory        
        date_flag = datetime.now().strftime("%Y-%m-%d_%H-%M")
        subdir_train = dir_train / f"EXP_ratio_f{fold_number}.base_{date_flag}"

        # run the base model
        log.info("Training the complete model...")
        data, model, checkpoint = train_model(
            directory=subdir_train, label="target", 
            epoch_max=tra_maxepoch, epoch_patience=tra_patience,
            dm=train_dm, arch=arch)
        
        results = pd.concat([base_results(dataset, fold_number, arch, False, random_state), 
                         data], axis = 1)
        results["nsamp_tra"] = len(train_dm.ds_train) + len(train_dm.ds_val)
        results["nsamp_pre"] = 0
        results["nsamp_test"] = len(train_dm.ds_test) 

        # update results file
        runs.append(results)
        log.info(f"Updating results file ({str(res_file)})")
        runs_df = pd.concat(runs, ignore_index=True)
        runs_df.to_csv(res_file, index=False)

        for j, pct_av_pre in enumerate(PCTS):

            # set the pretrain data ratio
            pretrain_dm.ds_train.frac_available = pct_av_pre
            pretrain_dm.ds_val.frac_available = pct_av_pre
            tot_pretrain_samps = len(pretrain_dm.ds_train) + len(pretrain_dm.ds_val)
            crun += 1 

            log.info(f"~ [{crun}/{trun}] Checking with {pct_av_pre*100}% pretrain data = {tot_pretrain_samps} ")

            # define the training directory   
            date_flag = datetime.now().strftime("%Y-%m-%d_%H-%M")
            subdir_train = dir_train / f"EXP_ratio_f{fold_number}.{i}_{date_flag}"

            results = base_results(dataset, fold_number, arch, True, random_state)
            results["nsamp_tra"] = len(train_dm.ds_train) + len(train_dm.ds_val)
            results["nsamp_pre"] = len(pretrain_dm.ds_train) + len(pretrain_dm.ds_val)
            results["nsamp_test"] = len(train_dm.ds_test)

            # pretrain the encoder
            log.info("Training the encoder...")
            data, model, checkpoint = train_model(directory=subdir_train, label="pretrain", 
                epoch_max=pre_maxepoch, epoch_patience=pre_patience,
                dm=pretrain_dm, arch=arch)
            results = pd.concat([results, data], axis=1)
            encoder = model.encoder

            # train with the original task
            log.info("Training the complete model...")
            data, model, checkpoint = train_model(directory=subdir_train, label="target", 
                epoch_max=tra_maxepoch, epoch_patience=tra_patience,
                dm=train_dm, arch=arch, encoder=encoder)
            results = pd.concat([results, data], axis=1)

            # update results file
            runs.append(results)
            log.info(f"Updating results file ({str(res_file)})")
            runs_df = pd.concat(runs, ignore_index=True)
            runs_df.to_csv(res_file, index=False)
            
    log.info(f"~~ EXPERIMENT COMPLETE! (fold #{fold_number+1}/{total_folds}) ~~")

    return runs_df

# =====================================================

def EXP_frames(
    dataset: str, arch: type[LightningModule],
    X_train: np.ndarray,  X_test: np.ndarray, 
    Y_train: np.ndarray,  Y_test: np.ndarray,
    fold_number: int = 0, random_state: int = 0,
    ):

    nsamp_tra: int = None 
    nsamp_pre: int = None
    nsamp_test: int = None

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