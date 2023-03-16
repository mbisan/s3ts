#/usr/bin/python3

""" Experiment to check the effect of train/pretrain sample ratios."""

from s3ts.experiments.common import create_folders, train_model, prepare_dms, base_results
from s3ts.data.modules import FullDataModule

from pytorch_lightning import LightningModule, seed_everything

import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path
import logging as log

def EXP_ratio(
    dataset: str, arch: type[LightningModule],
    X_train: np.ndarray,  X_test: np.ndarray, 
    Y_train: np.ndarray,  Y_test: np.ndarray,
    fold_number: int,
    total_folds: int, 
    rho_dfs: float = 0.1,
    quant_intervals: int = 5, 
    quant_shifts: list[int] = [0],
    batch_size: int = 128, 
    window_length: int = 5,
    window_stride: int = 1,
    random_state: int = 0,
    pre_maxepoch: int = 60,
    tra_maxepoch: int = 120,
    nsamp_tra: float = None, 
    nsamp_pre: float = None, 
    nsamp_test: float = None,
    learning_rate: float = 1e-4,
    dir_cache: Path = Path("cache/"),
    dir_train: Path = Path("training/"),
    dir_results: Path = Path("results/"),
    ) -> pd.DataFrame:

    exp_name = "ratio"
    log.info(f"~~ BEGIN '{exp_name}' EXPERIMENT (fold #{fold_number+1}/{total_folds}) ~~")

    # make sure folders exist
    create_folders() 
    res_file = dir_results / f"EXP_{exp_name}_{arch.__str__()}_{dataset}_f{fold_number}.csv"

    # NOTE: this is chosen so that the final number of
    # samples for just train and test is the same (50/50 split w/out pretrain)
    pret_frac = 1 - 1/(total_folds-1) 

    # prepare the data
    log.info("Preparing data modules...")
    train_dm, pretrain_dm = prepare_dms(dataset=dataset,
        X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
        batch_size=batch_size, window_length=window_length, window_stride=window_stride, 
        rho_dfs=rho_dfs, pret_frac=pret_frac, 
        quant_shifts=quant_shifts, quant_intervals=quant_intervals,
        nsamp_tra=nsamp_tra, nsamp_pre=nsamp_pre, nsamp_test=nsamp_test,
        fold_number=fold_number, random_state=random_state, 
        frames=arch.__frames__(), dir_cache=dir_cache)
    train_dm: FullDataModule
    pretrain_dm: FullDataModule

    runs = []
    PCTS = [0.2, 0.4, 0.6, 0.8, 1]
    PCTS = [1]
    trun, crun = len(PCTS)*(1+len(PCTS)), 0
    for i, pct_av_train in enumerate(PCTS):

        # reset the seed
        seed_everything(random_state)

        # set the train data ratio
        train_dm.ds_train.frac_available = pct_av_train
        train_dm.ds_val.frac_available = pct_av_train
        tot_train_samps = len(train_dm.ds_train) + len(train_dm.ds_val)

        crun += 1
        log.info(f"~ [{crun}/{trun}] Training baseline model... ({pct_av_train*100}% data = {tot_train_samps} samples)")

        # define the training directory        
        date_flag = datetime.now().strftime("%Y-%m-%d_%H-%M")
        subdir_train = dir_train / f"{date_flag}_EXP_{exp_name}_{arch.__str__()}_{dataset}_f{fold_number}-base"

        # run the base model
        log.info("Training the complete model...")
        data, model, checkpoint = train_model(
            directory=subdir_train, label="target", 
            epoch_max=tra_maxepoch, target="cls",
            dm=train_dm, arch=arch,
            learning_rate=learning_rate)
        
        results = pd.concat([base_results(dataset=dataset, fold_number=fold_number, arch=arch, pretrained=False, 
                            batch_size=batch_size, window_length=window_length, window_stride=window_stride,
                            random_state=random_state), 
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
            subdir_train = dir_train / f"{date_flag}_EXP_{exp_name}_{arch.__str__()}_{dataset}_f{fold_number}-{i}"
            
            results = base_results(dataset=dataset, fold_number=fold_number, arch=arch, pretrained=True, 
                        batch_size=batch_size, window_length=window_length, window_stride=window_stride,
                        random_state=random_state)
            results["nsamp_tra"] = len(train_dm.ds_train) + len(train_dm.ds_val)
            results["nsamp_pre"] = len(pretrain_dm.ds_train) + len(pretrain_dm.ds_val)
            results["nsamp_test"] = len(train_dm.ds_test)

            # reset the seed
            seed_everything(random_state)

            # pretrain the encoder
            log.info("Training the encoder...")
            data, model, checkpoint = train_model(
                directory=subdir_train, label="pretrain", 
                epoch_max=pre_maxepoch, target="reg",
                dm=pretrain_dm, arch=arch,
                learning_rate=learning_rate)
            results = pd.concat([results, data], axis=1)
            encoder = model.encoder

            # train with the original task
            log.info("Training the complete model...")
            data, model, checkpoint = train_model(
                directory=subdir_train, label="target", 
                epoch_max=tra_maxepoch, target="cls",
                dm=train_dm, arch=arch, 
                learning_rate=learning_rate,
                encoder=encoder)
            results = pd.concat([results, data], axis=1)

            # update results file
            runs.append(results)
            log.info(f"Updating results file ({str(res_file)})")
            runs_df = pd.concat(runs, ignore_index=True)
            runs_df.to_csv(res_file, index=False)
            
    log.info(f"~~ EXPERIMENT COMPLETE! (fold #{fold_number+1}/{total_folds}) ~~")

    return runs_df