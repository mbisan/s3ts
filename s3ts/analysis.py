#/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Helper functions for analysis of the data.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def load_folder(folder: Path) -> pd.DataFrame:

    """
        Load all csv files in a folder into a single dataframe.
        
        Parameters
        ----------
        folder : Path
            Path to the folder containing the csv files.
        
        Returns
        -------
        df : pd.DataFrame
            Dataframe containing all the data from the csv files.
    """

    # load all csv files in the folder using pandas
    df = pd.concat([pd.read_csv(f) for f in folder.glob("*.csv")])

    # fix missing values due to the way the data was saved
    df["stride_series"].replace(np.NaN, False, inplace=True)
    df['train_exc_limit'].replace(np.NaN, 0, inplace=True)
    df["pretrained"].replace(np.NaN, False, inplace=True)
    df["pretrain_mode"].replace(np.NaN, False, inplace=True)
    df["window_time_stride"].replace(np.NaN, 1, inplace=True)
    df["window_patt_stride"].replace(np.NaN, 1, inplace=True)
    df["cv_rep"].replace(np.NaN, False, inplace=True)
    df["eq_wdw_length"] = df["window_length"]*df["window_time_stride"]

    # check for duplicate entries
    file_entries = len(df)
    df = df.groupby(['mode', 'arch', 'dataset', 'pretrain_mode', 'window_length', "stride_series",
                    'window_time_stride', 'window_patt_stride', 'train_exc_limit', 'pretrained', "cv_rep"]).first().reset_index()
    unique_entries = len(df)
    print(f"{file_entries - unique_entries} duplicate entries removed")
        
    return df


def timedil_filter(df: pd.DataFrame) -> pd.DataFrame:

    # Filter the data
    data = df[df["pretrain_mode"] == False].copy()

    data = data[data['train_exc_limit'] == 32]
    data = data[data["window_patt_stride"] == 1]
    data = data[data["pretrained"] == False]

    dl_data = data[~data["arch"].isin(["nn"])]
    nn_data = data[~data["arch"].isin(["nn"])]
    
    data.sort_values(by=['dataset', "arch"], inplace=True)

    return dl_data, nn_data


def results_table(df: pd.DataFrame) -> pd.DataFrame:

    # Generate a table
    tab1 = data.groupby(["mode", "arch", "dataset", "eq_wdw_length"])["test_auroc"].mean().reset_index()
    tab2 = data.groupby(["mode", "arch", "dataset", "eq_wdw_length"])["test_auroc"].std().reset_index()
    tab1 = tab1[tab1["eq_wdw_length"] == 70].drop(columns=["eq_wdw_length"])
    tab2 = tab2[tab2["eq_wdw_length"] == 70].drop(columns=["eq_wdw_length"])
    tab = pd.merge(tab1, tab2, on=["dataset", "mode", "arch"]).rename(columns={"test_auroc_x": "mean", "test_auroc_y": "std"})
    tab["mean"], tab["std"] = tab["mean"]*100, tab["std"]*100
    tab.sort_values(by=["mode", "arch", "dataset"], inplace=True)
    

    return tab

