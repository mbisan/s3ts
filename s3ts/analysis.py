
from pathlib import Path
import pandas as pd
import numpy as np

def load_data(folder: Path) -> pd.DataFrame:

    """ Loads data form CSVs in a given folder. """

    dfs = []
    for file in folder.glob('*'):
        if ".csv" in file.name:
            dfs.append(pd.read_csv(file))
    df = pd.concat(dfs, ignore_index=True)

    # TODO grab nepochs from path to best model

    return df


def fix_close_values(df: pd.DataFrame, col: str)-> pd.DataFrame:

    """ Fix for the slight difference in the number of samples 
        that may appear due to rounding data splits. """

    unq = df[col].unique()
    df = df.copy()

    good_ones = [0]
    thresh = df[col].max()/10
    for i in unq:
        if np.all(np.array([np.abs(i-j) > thresh for j in good_ones])):
            good_ones.append(i)

    closest = []
    for i in unq:
        closest.append(np.argmin(np.abs(np.array(good_ones) - i)))

    mapping = {}
    for i, ns in enumerate(unq):
        mapping[ns] = unq[closest[i]]
    
    df[col] = df[col].replace(to_replace=mapping).copy()
    
    return df

def EXP_ratio_preprocessing(df: pd.DataFrame):

    """ Preprocess the data for the "ratio" experiment """

    cols = [c for c in df.columns if (("target_val" in c) or("target_test" in c) or (("pretrain_val" in c)) and "nepoch" not in c)]

    dfs = []
    for (dataset, arch), dfg in df.groupby(["dataset", "arch"]):

        if dataset in ["Coffee"]:
            dfg = fix_close_values(dfg, "nsamp_pre")

        df1 = dfg.groupby(["nsamp_tra", "nsamp_pre"])[cols].mean()
        df1.columns = [c + "_mean" for c in df1.columns]

        df2 = dfg.groupby(["nsamp_tra", "nsamp_pre"])[cols].std()
        df2.columns = [c + "_std" for c in df2.columns]

        pdf = pd.concat([df1, df2], axis=1).reset_index()
        pdf = pdf[pdf.columns.sort_values()].copy()
        pdf.insert(0, column="dataset", value=dataset)
        pdf.insert(0, column="arch", value=arch)
        dfs.append(pdf)

    df = pd.concat(dfs, ignore_index=True)


    return df

def EXP_ratio_set_baselines(df: pd.DataFrame):

    df = df.copy()

    dfs = []
    df.drop(columns=[c for c in df.columns if ("pretrain" in c)], inplace=True)
    mcols = [c for c in df.columns if ("target_" in c) and ("_mean" in c)]
    ecols = [c for c in df.columns if ("target_" in c) and ("_std" in c)]
    for xd, dfg in df.groupby(["arch", "dataset", "nsamp_tra"]):
        if (dfg["nsamp_pre"] > 0).sum() > 0:
            dfg.loc[dfg["nsamp_pre"] > 0, mcols] = dfg.loc[dfg["nsamp_pre"] > 0, mcols] - dfg.iloc[0][mcols]
            dfg.loc[dfg["nsamp_pre"] > 0, ecols] = dfg.loc[dfg["nsamp_pre"] > 0, ecols] + dfg.iloc[0][ecols]
            dfs.append(dfg.iloc[1:].copy())
        else:
            continue

    return pd.concat(dfs, ignore_index=True)
    

