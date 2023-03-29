#/usr/bin/python3

""" Experiment to check the effect of train/pretrain sample ratios."""

from s3ts.experiments.common import prepare_dm, train_model, update_results_file 
from s3ts.data.modules import DFDataModule

import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path
import logging as log

def EXP_ratio(
    dataset: str, repr: str, arch: str,
    X_train: np.ndarray,  X_pretest: np.ndarray, 
    Y_train: np.ndarray,  Y_pretest: np.ndarray,
    fold_number: int, total_folds: int, 
    rho_dfs: float,
    batch_size: int, val_size: float,
    window_length: int, stride_series: bool,
    window_time_stride: int, window_patt_stride: int,
    train_events_per_class: int, train_event_multiplier: int,
    pret_event_multiplier: int, test_event_multiplier: int,
    max_epoch_pre: int = 60,
    max_epoch_tra: int = 120,
    learning_rate: float = 1e-4,
    random_state: int = 0,
    use_cache: bool = True,
    pattern_type: str = "medoids",
    cache_dir: Path = Path("cache/"),
    train_dir: Path = Path("training/"),
    results_dir: Path = Path("results/")
    ) -> pd.DataFrame:

    exp_name = "ratio"
    log.info(f"~~ BEGIN '{exp_name}' EXPERIMENT (fold #{fold_number+1}/{total_folds}) ~~")

    # Ensure folders exist
    for folder in [cache_dir, train_dir, results_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # define the training directory        
    date_flag = datetime.now().strftime("%Y-%m-%d_%H-%M")
    subdir_train = train_dir / f"{date_flag}_EXP_{exp_name}_{repr}_{arch}_{dataset}_f{fold_number}"

    # Output file for the results
    res_file = results_dir / f"EXP_{exp_name}_{repr}_{arch}_{dataset}_f{fold_number}.csv"

    # Experiment parameters    
    EVENTS_PER_CLASS = [8, 16, 32]
    PRET_MULTIPLIERS = [4, 8, 16]

    # Create the data module
    dm : DFDataModule = prepare_dm(dataset=dataset, rho_dfs=rho_dfs,
        X_train=X_train, X_pretest=X_pretest, Y_train=Y_train, Y_pretest=Y_pretest,
        train_event_multiplier=train_event_multiplier, train_events_per_class=max(EVENTS_PER_CLASS), 
        pret_event_multiplier=max(PRET_MULTIPLIERS), test_event_multiplier=test_event_multiplier, 
        pattern_type=pattern_type, batch_size=batch_size, val_size=val_size, 
        window_length=window_length, stride_series=stride_series,
        window_time_stride=window_time_stride, window_patt_stride=window_patt_stride,
        fold_number=fold_number, random_state=random_state, cache_dir=cache_dir, use_cache=use_cache)

    runs: list[pd.DataFrame] = []
    for i, exc in enumerate(EVENTS_PER_CLASS):

        # Update the data module
        dm.update_properties(av_train_events=exc*dm.n_classes*train_event_multiplier)

        # Run the model without pretraining
        data, _ = train_model(
            dataset=dataset, repr=repr, arch=arch, dm=dm, pretrain=False, 
            fold_number=fold_number, directory=subdir_train, label=f"{exc}exc_base",
            max_epoch_pre = max_epoch_pre, max_epoch_tgt=max_epoch_tra,
            learning_rate=learning_rate, random_state=random_state,
            train_events_per_class=exc, train_event_multiplier=train_event_multiplier,
            pret_event_multiplier=0, test_event_multiplier=test_event_multiplier)
        
        # Update results file
        runs = update_results_file(res_list=runs, new_res=data, res_file=res_file)

        for j, pmult in enumerate(PRET_MULTIPLIERS):

            # Update the data module
            dm.update_properties(av_train_events=exc*dm.n_classes*train_event_multiplier,
                av_pret_events=max(EVENTS_PER_CLASS)*dm.n_classes*pmult)

            # Run the model with pretraining 
            data, _ = train_model(
                dataset=dataset, repr=repr, arch=arch, dm=dm, pretrain=True, 
                fold_number=fold_number, directory=subdir_train, label=f"{exc}exc_{pmult}pmult",
                max_epoch_pre = max_epoch_pre, max_epoch_tgt=max_epoch_tra,
                learning_rate=learning_rate, random_state=random_state,
                train_events_per_class=exc, train_event_multiplier=train_event_multiplier,
                pret_event_multiplier=pmult, test_event_multiplier=test_event_multiplier)
            
            # Update results file
            runs = update_results_file(res_list=runs, new_res=data, res_file=res_file)

    log.info(f"~~ EXPERIMENT COMPLETE! (fold #{fold_number+1}/{total_folds}) ~~")

    # Return the results data
    return pd.concat(runs, ignore_index=True)