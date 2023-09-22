#/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def timedil_figure(
        df: pd.DataFrame,
        fontsize: int = 18,
        metric: str = "test_acc",
        fname: str = "figures/ablation/timedil_new.pdf",
        ) -> None:
    
    """ Generates the time dilation figure for the paper. """

    # cleanup

    data = df[df["pretrain_mode"] == False].copy()
    data = data[data['train_exc_limit'] == 32]
    data = data[data["window_patt_stride"] == 1]
    data = data[data["pretrained"] == False]
    data = data[~data["mode"].isin(["ts"])]

    metric: str = "test_acc"

    data[metric] = data[metric]*100

    data["Method"] =  data["arch"] + "_" + data["mode"]
    data.sort_values(["Method"], inplace=True)

    data["arch"].replace(to_replace=["rnn", "cnn", "res"], value=["RNN", "CNN", "RES"], inplace=True)
    data["mode"].replace(to_replace=["df", "gf"], value=["DF", "GF"], inplace=True)
    data["Arch"] = data["arch"]

    # aggregation 

    dfs = []
    for g, gdf in data.groupby(["mode", "arch", "dataset"]):

        bline = gdf.groupby(["mode", "arch", "dataset", "eq_wdw_length"])[metric].mean().iloc[0]
        gdf[metric] = (gdf[metric] - bline)/bline

        mean = gdf.groupby(["mode", "arch", "dataset", "eq_wdw_length"])[metric].mean()
        mean.name = "mean"
        std  = gdf.groupby(["mode", "arch", "dataset", "eq_wdw_length"])[metric].std().rename({metric: "std"})
        std.name = "std"
        
        gdat = pd.concat([mean, std], axis=1).reset_index()
        dfs.append(gdat)
    pdata = pd.concat(dfs)



    plt.rc('font', family='serif', serif='Times New Roman', size=fontsize)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')


    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,6), gridspec_kw={"wspace": 0} )
    ax: list[plt.Axes] 

    cmap = plt.get_cmap("tab10", pdata["dataset"].nunique())
    colors =  {dset: cmap(i) for i, dset in enumerate(pdata["dataset"].unique())}
    nums =  {dset: i+1 for i, dset in enumerate(pdata["dataset"].unique())}


    for (mode, arch, dset), gdf in pdata.groupby(["mode", "arch", "dataset"]):

        ax_idx = 0 if mode == "DF" else 1
        ltype = "solid" if arch == "CNN" else "dashed"
        color = colors[dset]
        iax = ax[ax_idx]
        if ax_idx == 0:
            label = str(nums[dset]) if arch == "CNN" else None
        else:
            label = str(nums[dset]) if arch == "RES" else None
            iax.set_yticklabels([])
        iax.set_yticks([-30,-20,-10,0,10,20,30,40,50,60])
        iax.set_xticks([1,3,5,7])
        iax.set_title(mode)
        iax.set_ylim(-40, 70)
        iax.grid(True, axis="y")
        iax.errorbar(gdf["eq_wdw_length"]//10, gdf["mean"]*100, yerr=gdf["std"]*100, linestyle=ltype, label=label, c=color)

    # legends
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="center", bbox_to_anchor=(1.02,0.65), borderpad=0.8, title="CNN", ncols=2, fancybox=False, shadow=True)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="center", bbox_to_anchor=(1.02,0.35), borderpad=0.8, title="RES", ncols=2, fancybox=False, shadow=True)

    # x label
    fig.text(0.5, 0, '$\delta$', horizontalalignment='center', verticalalignment='center', transform=fig.transFigure)
    # y label
    ax[0].set_ylabel(r"% Change in Test Accuracy");

    plt.savefig(fname, bbox_inches='tight')
