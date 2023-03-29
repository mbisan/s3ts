#/usr/bin/python3

""" Experiment to check the effect of frame stride."""

from s3ts.experiments.common import prepare_dm, train_model, update_results_file 
from s3ts.data.modules import DFDataModule

import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path
import logging as log

def EXP_wdw(
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

    exp_name = "wdw"
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
    WINDOW_LENGTHS = [5, 10, 15]
    TIME_STRIDES = [2, 3, 4, 5]
    PATT_STRIDES = [2, 3, 4, 5]

    WINDOW_LENGTHS = [5]
    TIME_STRIDES = [2]
    PATT_STRIDES = [2]

    # Create the data module
    dm : DFDataModule = prepare_dm(dataset=dataset, rho_dfs=rho_dfs,
        X_train=X_train, X_pretest=X_pretest, Y_train=Y_train, Y_pretest=Y_pretest,
        train_event_multiplier=train_event_multiplier, train_events_per_class=train_events_per_class, 
        pret_event_multiplier=pret_event_multiplier, test_event_multiplier=test_event_multiplier, 
        pattern_type=pattern_type, batch_size=batch_size, val_size=val_size, 
        window_length=window_length, stride_series=stride_series,
        window_time_stride=window_time_stride, window_patt_stride=window_patt_stride,
        fold_number=fold_number, random_state=random_state, cache_dir=cache_dir, use_cache=use_cache)

    runs: list[pd.DataFrame] = []

    # Run the model with different window lengths
    log.info(f"~~ Running the training with different window sizes ~~")
    for i, wlen in enumerate(WINDOW_LENGTHS):

        # Update the data module
        dm.update_properties(window_length=wlen)

        # Run the model without pretraining
        data, _ = train_model(
            label=f"wdw{wlen}_wts{window_time_stride}_wps{window_patt_stride}",
            dataset=dataset, repr=repr, arch=arch, dm=dm, pretrain=False, 
            fold_number=fold_number, directory=subdir_train, 
            max_epoch_pre = max_epoch_pre, max_epoch_tgt=max_epoch_tra,
            learning_rate=learning_rate, random_state=random_state,
            train_events_per_class=train_events_per_class, 
            train_event_multiplier=train_event_multiplier,
            test_event_multiplier=test_event_multiplier,
            pret_event_multiplier=0)
        
        # Update results file
        runs = update_results_file(res_list=runs, new_res=data, res_file=res_file)

        # Run the model with pretraining 
        data, _ = train_model(
            label=f"wdw{wlen}_wts{window_time_stride}_wps{window_patt_stride}_pre",
            dataset=dataset, repr=repr, arch=arch, dm=dm, pretrain=True, 
            fold_number=fold_number, directory=subdir_train,
            max_epoch_pre = max_epoch_pre, max_epoch_tgt=max_epoch_tra,
            learning_rate=learning_rate, random_state=random_state,
            train_events_per_class=train_events_per_class, 
            train_event_multiplier=train_event_multiplier,
            test_event_multiplier=test_event_multiplier,
            pret_event_multiplier=pret_event_multiplier)
        
        # Update results file
        runs = update_results_file(res_list=runs, new_res=data, res_file=res_file)

    # Reset the data module
    dm.update_properties(window_length=window_length)

    # Run the model with different window time strides
    log.info(f"~~ Running the training with different window time strides ~~")
    for wts in TIME_STRIDES:

        # Update the data module
        dm.update_properties(window_time_stride=wts, stride_series=True)

        # Run the model without pretraining
        data, _ = train_model(
            label=f"wdw_{window_length}_wts{wts}_wps{window_patt_stride}",
            dataset=dataset, repr=repr, arch=arch, dm=dm, pretrain=False, 
            fold_number=fold_number, directory=subdir_train, 
            max_epoch_pre = max_epoch_pre, max_epoch_tgt=max_epoch_tra,
            learning_rate=learning_rate, random_state=random_state,
            train_events_per_class=train_events_per_class, 
            train_event_multiplier=train_event_multiplier,
            test_event_multiplier=test_event_multiplier,
            pret_event_multiplier=0)
        
        # Update results file
        runs = update_results_file(res_list=runs, new_res=data, res_file=res_file)

        # Run the model with pretraining and striding the TS
        data, _ = train_model(
            label=f"wdw_{window_length}_ts{wts}_ps{window_patt_stride}_pre_st",
            dataset=dataset, repr=repr, arch=arch, dm=dm, pretrain=True, 
            fold_number=fold_number, directory=subdir_train,
            max_epoch_pre = max_epoch_pre, max_epoch_tgt=max_epoch_tra,
            learning_rate=learning_rate, random_state=random_state,
            train_events_per_class=train_events_per_class, 
            train_event_multiplier=train_event_multiplier,
            test_event_multiplier=test_event_multiplier,
            pret_event_multiplier=pret_event_multiplier)
        
        # Update the data module
        dm.update_properties(stride_series=False)

        # Update results file
        runs = update_results_file(res_list=runs, new_res=data, res_file=res_file)
   
        # Run the model with pretraining and striding the TS
        data, _ = train_model(
            label=f"wdw{window_length}_wts{wts}_wps{window_patt_stride}_pre_nst",
            dataset=dataset, repr=repr, arch=arch, dm=dm, pretrain=True, 
            fold_number=fold_number, directory=subdir_train,
            max_epoch_pre = max_epoch_pre, max_epoch_tgt=max_epoch_tra,
            learning_rate=learning_rate, random_state=random_state,
            train_events_per_class=train_events_per_class, 
            train_event_multiplier=train_event_multiplier,
            test_event_multiplier=test_event_multiplier,
            pret_event_multiplier=pret_event_multiplier)
        
        # Update results file
        runs = update_results_file(res_list=runs, new_res=data, res_file=res_file)
   
    # Reset the data module
    dm.update_properties(window_time_stride=window_time_stride, 
        stride_series=stride_series)

    # Run the model with different window pattern strides
    log.info(f"~~ Running the training with different window pattern strides ~~")
    for wps in PATT_STRIDES:

        # Update the data module
        dm.update_properties(window_patt_stride=wps)

        # Run the model without pretraining
        data, _ = train_model(
            label=f"wlen{window_length}_wts{window_time_stride}_wps{wps}",
            dataset=dataset, repr=repr, arch=arch, dm=dm, pretrain=False, 
            fold_number=fold_number, directory=subdir_train, 
            max_epoch_pre = max_epoch_pre, max_epoch_tgt=max_epoch_tra,
            learning_rate=learning_rate, random_state=random_state,
            train_events_per_class=train_events_per_class, 
            train_event_multiplier=train_event_multiplier,
            test_event_multiplier=test_event_multiplier,
            pret_event_multiplier=0)
        
        # Update results file
        runs = update_results_file(res_list=runs, new_res=data, res_file=res_file)

        # Run the model with pretraining and striding the TS
        data, _ = train_model(
            label=f"wlen{window_length}_wts{window_time_stride}_wps{wps}_pre",
            dataset=dataset, repr=repr, arch=arch, dm=dm, pretrain=True, 
            fold_number=fold_number, directory=subdir_train,
            max_epoch_pre = max_epoch_pre, max_epoch_tgt=max_epoch_tra,
            learning_rate=learning_rate, random_state=random_state,
            train_events_per_class=train_events_per_class, 
            train_event_multiplier=train_event_multiplier,
            test_event_multiplier=test_event_multiplier,
            pret_event_multiplier=pret_event_multiplier)
        
        # Update the data module
        dm.update_properties(stride_series=False)

        # Update results file
        runs = update_results_file(res_list=runs, new_res=data, res_file=res_file)

    log.info(f"~~ EXPERIMENT COMPLETE! (fold #{fold_number+1}/{total_folds}) ~~")

    # Return the results data
    return pd.concat(runs, ignore_index=True)