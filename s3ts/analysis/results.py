#/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

def load_folder(folder: Path) -> pd.DataFrame:

    """ Load all csv files in the results folder into a single dataframe. """

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
    print(f"{len(df)} total entries")
        
    return df

def results_table(df: pd.DataFrame, 
        metric: str = "test_acc",
        ) -> pd.DataFrame:

    """ Generates the results table for the paper. """

    data = df[df["pretrain_mode"] == False].copy()
    data = data[
        (data["arch"] == "nn") |
        ((data["mode"] == "ts") & (data["window_length"] == 70)) |
        ((data["train_exc_limit"] == 32) & (data["window_length"] == 10) & (data["window_time_stride"] == 7) & (data["window_patt_stride"] == 1))
    ]

    def method(row):
        string = row["mode"].upper() + "-" +  row["arch"].upper() 
        if row["pretrained"]:
            if row["stride_series"]:
                string += "-B"
            else:
                string += "-A"
        return string

    data["method"] = data[["arch", "mode", "pretrained", "stride_series"]].apply(method, axis=1)

    tab1 = (data.groupby(["method", "dataset"])[[metric]].mean()*100).reset_index()
    tab2 = (data.groupby(["method", "dataset"])[[metric]].std()*100).reset_index()
    tab1["var"], tab2["var"] = "mean", "std"

    table = pd.concat([tab1, tab2])
    table = table.groupby(["method", "var", "dataset"])[metric].mean().unstack().unstack().round(0)
    table["avg_rank"] = tab1.groupby(["method", "dataset"])[metric].mean().unstack().rank(ascending=False).mean(axis=1).round(1)

    return table

