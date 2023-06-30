#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Helper functions for analysis of the output data. """

# package imports
from s3ts.data.acquisition import download_dataset
from s3ts.data.setup import train_test_splits
from s3ts.data.setup import setup_train_dm 

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# standard
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
    table = table.groupby(["method", "var", "dataset"])[metric].mean().unstack().unstack().round(2)
    table["avg_rank"] = tab1.groupby(["method", "dataset"])[metric].mean().unstack().rank(ascending=False).mean(axis=1).round(2)

    return table


def timedil_figure(
        df: pd.DataFrame,
        fontsize: int = 18,
        metric: str = "test_acc",
        metric_lab: str = "Test Accuracy",
        fname: str = "figures/ablation/timedil.pdf",
        ) -> pd.DataFrame:
    
    """ Generates the time dilation figure for the paper. """

    # Filter the data
    data = df[df["pretrain_mode"] == False].copy()
    data = data[data['train_exc_limit'] == 32]
    data = data[data["window_patt_stride"] == 1]
    data = data[data["pretrained"] == False]
    data = data[~data["mode"].isin(["ts"])]
    
    # Convert metric to percentage
    data[metric] = data[metric]*100

    # Generate a plot
    data["Method"] =  data["arch"] + "_" + data["mode"]
    data.sort_values(["Method"], inplace=True)

    data["arch"].replace(to_replace=["rnn", "cnn", "res"], value=["RNN", "CNN", "RES"], inplace=True)
    data["mode"].replace(to_replace=["df", "gf"], value=["DF", "GAF"], inplace=True)
    data["Arch"] = data["arch"]

    sns.set_theme()
    sns.set_style("whitegrid")
    plt.rc('font', family='serif', serif='Times New Roman', size=fontsize)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')

    g: sns.FacetGrid = sns.relplot(data=data, x="eq_wdw_length", y=metric, hue='Arch', 
        kind="line", col="mode", row="dataset", palette=sns.color_palette("bright"),
        height=2.5, aspect=1.25, legend="auto", markers="True", col_order=["DF", "GAF"], 
        row_order=["ArrowHead", "CBF", "ECG200", "GunPoint", "SyntheticControl", "Trace"],
        facet_kws={"despine": False, "margin_titles": True, "legend_out": True,
                   "sharey": False, "sharex": True})
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=fontsize)
    g.set_xlabels("Equivalent Window Length", fontsize=fontsize-2);
    g.set_ylabels(metric_lab, fontsize=fontsize-2);
    g.set(xlim=(5, 75), xticks=[10, 30, 50, 70])
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.legend.set_title("")

    dsets = ["ArrowHead", "CBF", "ECG200", "GunPoint", "SyntheticControl", "Trace"]
    ybounds = {
        "ArrowHead": ([35, 60], [40, 45, 50, 55]),
        "CBF": ([35, 75], [40, 50, 60, 70]),
        "ECG200": ([45, 70], [50, 55, 60, 65]),
        "GunPoint": ([45, 70], [50, 55, 60, 65]),
        "SyntheticControl": ([25, 65], [30, 40, 50, 60]),
        "Trace": ([45, 70], [50, 55, 60, 65])}

    for i, row in enumerate(g.axes):
        for j, ax in enumerate(row):
            ax.set_ylim(ybounds[dsets[i]][0])
            ax.set_yticks(ybounds[dsets[i]][1])

    for (row_val, col_val), ax in g.axes_dict.items():
        if col_val in ["GAF"]:
            ax.set_yticklabels([])
        if row_val not in ["CBF", "SyntheticControl"]:
            ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_color("black")
        if row_val == "Trace":
            ax.set_xlabel("")
            if col_val == "DF":
                ax.text(1, -0.32, "Context Size", ha="center", va="center", transform=ax.transAxes, fontsize=fontsize-2)


    g.savefig(fname, bbox_inches='tight')

def pretrain_figure(df: pd.DataFrame,
        fontsize: int = 18,
        metric: str = "test_acc",
        metric_lab: str = "Test Accuracy",
        fname: str = "figures/ablation/pretrain.pdf"
        ) -> pd.DataFrame:

    data = df[df["pretrain_mode"] == False].copy()

    data = data[data["window_time_stride"] == 7]
    data = data[data["window_patt_stride"] == 1]

    data = data[data["arch"].isin(["cnn", "res"])]

    # Convert metric to percentage
    data[metric] = data[metric]*100

    data["arch"].replace(to_replace=["cnn", "res"], value=["CNN", "RES"], inplace=True)
    data["mode"].replace(to_replace=["df", "gf"], value=["DF", "GAF"], inplace=True)
    data["Arch"] = data["arch"] + data["pretrained"].replace({True: "-", False: ""}) + data["stride_series"].replace({True: "B", False: "A"})
    data["Arch"].replace({"CNNA": "CNN", "RESA": "RES"}, inplace=True)

    sns.set_theme()
    sns.set_style("whitegrid")
    plt.rc('font', family='serif', serif='Times New Roman', size=fontsize)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')

    g: sns.FacetGrid = sns.relplot(data=data, x="train_exc_limit", y=metric, hue='Arch', 
        kind="line", col="mode", row="dataset", palette=sns.color_palette("bright"),
        height=2.5, aspect=1.25, legend="brief", markers="True", col_order=["DF", "GAF"], 
        row_order=["ArrowHead", "CBF", "ECG200", "GunPoint", "SyntheticControl", "Trace"],
        facet_kws={"despine": False, "margin_titles": True, "legend_out": True,
                   "sharey": False, "sharex": True})

    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=fontsize)
    g.set_xlabels("Labeled events per class", fontsize=fontsize-2);
    g.set_ylabels(metric_lab, fontsize=fontsize-2);
    g.set(xlim=(2, 34), xticks=[8,16,24,32])
    #    ylim=[15, 80], yticks=[20, 30, 40, 50, 60, 70])
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.legend.set_bbox_to_anchor((1.020, 0.5))
    g.legend.set_title("")

    dsets = ["ArrowHead", "CBF", "ECG200", "GunPoint", "SyntheticControl", "Trace"]
    ybounds = {
        "ArrowHead": ([35, 60], [40, 45, 50, 55]),
        "CBF": ([40, 80], [50, 60, 70]),
        "ECG200": ([45, 70], [50, 55, 60, 65]),
        "GunPoint": ([45, 70], [50, 55, 60, 65]),
        "SyntheticControl": ([15, 55], [20, 30, 40, 50]),
        "Trace": ([45, 80], [50, 55, 60, 65])}

    for i, row in enumerate(g.axes):
        for j, ax in enumerate(row):
            ax.set_ylim(ybounds[dsets[i]][0])
            ax.set_yticks(ybounds[dsets[i]][1])
    for (row_val, col_val), ax in g.axes_dict.items():
        if col_val in ["GAF"]:
            ax.set_yticklabels([])
        if row_val not in ["CBF", "SyntheticControl"]:
            ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_color("black")
        if row_val == "Trace":
            ax.set_xlabel("")
            if col_val == "DF":
                ax.text(1, -0.32, "Labeled events per class", ha="center", va="center", transform=ax.transAxes, fontsize=fontsize-2)

    g.savefig(fname, bbox_inches='tight')

def encoding_plot(dataset: str,
        folder: Path = Path("figures/encodings"),
        ) -> None:
    
    """ Generates the encoding plots of a given dataset. """

    # dataset settings
    cv_rep          = 0
    exc             = 32
    random_state    = 0
    rho_dfs         = 0.1

    # plot settings
    shift, n_frames = 0, 6
    vmin, vmax = -2.5, 2.5
    print_lines = False
    fontsize = 18
    ts_color = "darkslategray"
    cmaps = ["seismic", "seismic"]
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    # load the dataset
    X, Y, medoids, medoid_idx = download_dataset(dataset=dataset, storage_dir=Path("storage"))
    for j, (train_idx, test_idx) in enumerate(train_test_splits(X, Y, exc=exc, nreps=cv_rep+1, random_state=random_state)):
                if j == cv_rep:
                    break

    # setup the datamodules
    dm_df = setup_train_dm(X=X, Y=Y, patterns=medoids, train_idx=train_idx, test_idx=test_idx,
        test_sts_length=200, train_event_mult=4, train_strat_size=2, train_event_limit=exc,
        batch_size=128, val_size=0.25, rho_dfs=rho_dfs, window_length=10, mode="df", num_workers=4,
        window_time_stride=7, window_patt_stride=1,stride_series=False, random_state=random_state)
    dm_gf = setup_train_dm(X=X, Y=Y, patterns=medoids, train_idx=train_idx, test_idx=test_idx,
        test_sts_length=200, train_event_mult=4, train_strat_size=2, train_event_limit=exc,
        batch_size=128, val_size=0.25, rho_dfs=rho_dfs, window_length=10, mode="gf", num_workers=4,
        window_time_stride=7, window_patt_stride=1,stride_series=False, random_state=random_state)

    # load the required data
    n_patterns = dm_df.n_patterns
    sts_range = (dm_df.l_events*shift, dm_df.l_events*shift + dm_df.l_events*n_frames)
    patts = dm_df.patterns.numpy()
    TS = dm_df.STS_train[sts_range[0]:sts_range[1]]
    DF = dm_df.DM_train[:,:,sts_range[0]:sts_range[1]]
    GF = dm_gf.DM_train[:,:,sts_range[0]:sts_range[1]]

    fig = plt.figure(figsize=(12, 1.5+n_patterns*1.5), dpi=100, layout="constrained")
    gs = fig.add_gridspec(nrows=n_patterns+1, ncols=3,
            hspace=0, height_ratios=None,
            wspace=0, width_ratios=[0.1, 0.45, 0.45])

    vlines = np.where(np.mod(np.arange(sts_range[0], sts_range[1]), dm_df.l_events) == 0)[0]

    corner = fig.add_subplot(gs[0,0])
    corner.set_xticklabels([]), corner.set_yticklabels([])
    corner.set_xticks([]), corner.set_yticks([])
    corner.text(0.95, 0.5, "STS", fontsize=fontsize, rotation=90, ha="center", va="center")
    corner.text(0.5, 0.05, "Patterns", fontsize=fontsize, ha="center", va="center")
    corner.spines['top'].set_visible(False)
    corner.spines['right'].set_visible(False)
    corner.spines['bottom'].set_visible(False)
    corner.spines['left'].set_visible(False)

    sts_ax1 = fig.add_subplot(gs[0,1])
    sts_ax1.plot(np.arange(sts_range[0], sts_range[1]), TS, color=ts_color)
    sts_ax1.set_xlim(sts_range[0], sts_range[1]-1)
    sts_ax1.set_xticklabels([]), sts_ax1.set_yticklabels([])
    sts_ax1.set_xticks([]), sts_ax1.set_yticks([])
    if print_lines:
            [sts_ax1.axvline(x + sts_range[0], color="dimgray", linewidth=0.5) for x in vlines]
    sts_ax1.grid(True)

    sts_ax2 = fig.add_subplot(gs[0,2])
    sts_ax2.plot(np.arange(sts_range[0], sts_range[1]), TS, color=ts_color)
    sts_ax2.set_xlim(sts_range[0], sts_range[1]-1)
    sts_ax2.set_xticklabels([]), sts_ax2.set_yticklabels([])
    sts_ax2.set_xticks([]), sts_ax2.set_yticks([])
    if print_lines:
            [sts_ax2.axvline(x + sts_range[0], color="dimgray", linewidth=0.5) for x in vlines]
    sts_ax1.grid(True)

    for p in range(n_patterns):

            # patt plot
            patt_ax = fig.add_subplot(gs[p+1,0])
            patt_ax.plot(patts[p,::-1], np.arange(len(patts[0])), color=ts_color)
            patt_ax.set_yticklabels([]), patt_ax.set_xticklabels([])
            patt_ax.set_yticks([]), patt_ax.set_xticks([])
            patt_ax.invert_xaxis(), patt_ax.grid(True)

            # df plot
            df_ax = fig.add_subplot(gs[p+1,1])        
            df_im = DF[p,:,sts_range[0]:sts_range[1]]
            df_im = 2*(df_im-df_im.mean())/df_im.std()
            df_ax.imshow(df_im, aspect="auto", cmap=cmaps[0], vmin=vmin, vmax=vmax)
            df_ax.set_yticklabels([]), df_ax.set_xticklabels([])
            df_ax.set_xticks([]), df_ax.set_yticks([])

            # gf plot
            gf_ax = fig.add_subplot(gs[p+1,2])
            gf_im = GF[p,:,sts_range[0]:sts_range[1]]
            gf_im = 2*(gf_im-gf_im.mean())/gf_im.std()
            gf_ax.imshow(gf_im, aspect="auto", cmap=cmaps[1], vmin=vmin, vmax=vmax)
            gf_ax.set_yticklabels([]), gf_ax.set_xticklabels([])
            gf_ax.set_xticks([]), gf_ax.set_yticks([])

            patt_ax.set_ylabel("Class {}".format(p+1), fontsize=fontsize)

            if p == n_patterns-1:
                    df_ax.set_xlabel("DF Representation", fontsize=fontsize)
                    gf_ax.set_xlabel("GAF Representation", fontsize=fontsize)

    plt.savefig(folder / f"enc_{dataset}.png", bbox_inches="tight")


if __name__ == "__main__":
     
    df = load_folder(Path("storage/synced"))

    # results table
    print(results_table(df))

    # timedil figure
    timedil_figure(df)

    # pretrain figure
    pretrain_figure(df)

    # encoding plots
    for dset in ["ArrowHead", "CBF", "ECG200", "GunPoint", "SyntheticControl", "Trace"]:
        encoding_plot(dset)

    plt.show()


