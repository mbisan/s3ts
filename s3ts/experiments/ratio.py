#/usr/bin/python3

""" Experiment to check the effect of train/pretrain sample ratios."""

from s3ts.experiments.common import prepare_dm, train_model, update_results_file 
from s3ts.data.modules import DFDataModule

from pytorch_lightning import LightningModule, seed_everything

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
    batch_size: int, window_length: int,
    window_time_stride: int, window_pattern_stride: int,
    train_samples_per_class: int, train_sample_multiplier: int,
    pret_sample_multiplier: int, test_sample_multiplier: int,
    pre_maxepoch: int = 60,
    tra_maxepoch: int = 120,
    learning_rate: float = 1e-4,
    random_state: int = 0,
    use_cache: bool = True,
    pattern_type = "medoids",
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
    subdir_train =train_dir / f"{date_flag}_EXP_{exp_name}_{arch.__str__()}_{dataset}_f{fold_number}"

    # Output file for the results
    res_file = results_dir / f"EXP_{exp_name}_{arch.__str__()}_{dataset}_f{fold_number}.csv"

    # Create the data module
    dm : DFDataModule = prepare_dm(dataset=dataset, rho_dfs=rho_dfs,
        X_train=X_train, X_pretest=X_pretest, Y_train=Y_train, Y_pretest=Y_pretest,
        train_sample_multiplier=train_sample_multiplier, train_samples_per_class=train_samples_per_class, 
        pret_sample_multiplier=pret_sample_multiplier, test_sample_multiplier=test_sample_multiplier, 
        pattern_type=pattern_type, batch_size=batch_size, window_length=window_length,
        window_time_stride=window_time_stride, window_pattern_stride=window_pattern_stride,
        fold_number=fold_number, random_state=random_state, cache_dir=cache_dir, use_cache=use_cache)

    runs: list[pd.DataFrame] = []
    EVENTS_PER_CLASS = [4, 8, 16]
    PRET_MULTIPLIERS = [1, 2, 4, 8, 16]

    for i, exc in enumerate(EVENTS_PER_CLASS):

        # Set the random seed
        seed_everything(random_state, workers=True)

        # Update the data module
        dm.update_sample_availability(av_train_samples=exc*dm.n_classes*train_sample_multiplier,
            av_pret_samples=exc*dm.n_classes*train_sample_multiplier*pret_sample_multiplier)

        # run the base model
        data, _ = train_model(
            directory=subdir_train, label="target", 
            epoch_max=tra_maxepoch, target="cls",
            dm=dm, arch=arch,
            learning_rate=learning_rate)
        
        # update results file
        runs = update_results_file(res_list=runs, new_res=data, res_file=res_file)

        for j, pmult in enumerate(PRET_MULTIPLIERS):
    
            # Set the random seed
            seed_everything(random_state, workers=True)

            # Update the data module
            dm.update_sample_availability(av_pret_samples=exc*dm.n_classes*train_sample_multiplier*pmult)
            
            data, _ = train_model(
                directory=subdir_train, label="target", 
                epoch_max=tra_maxepoch, target="cls",
                dm=dm, arch=arch,
                learning_rate=learning_rate)
            
    log.info(f"~~ EXPERIMENT COMPLETE! (fold #{fold_number+1}/{total_folds}) ~~")

    # Return the results data
    return pd.concat(runs, ignore_index=True)