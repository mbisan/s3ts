#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Data modules for the S3TS project. """

# torch / lightning imports
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch

# standard library imports
import multiprocessing as mp
import numpy as np

class StaticDS(Dataset):

    DM: torch.Tensor
    STS: torch.Tensor
    SCS: torch.Tensor
    index: np.ndarray
    wdw_len: int
    wdw_str: int
    sts_str: bool
    DM_transform: torch.nn.Module
    STS_transform: torch.nn.Module

    def __init__(self, DM, STS, SCS, wdw_len, wdw_str, sts_str,
            index=None, DM_trans=None, STS_trans=None) -> None:
        
        # save parameters as attributes
        super(StaticDS).__init__(), self.__dict__.update(locals())

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        # index
        idx = self.index[idx]
        # frame
        frame = self.DM[:,::self.wdw_str,idx - self.wdw_len*self.wdw_str+1:idx+1:self.wdw_str]
        if self.DM_transform:
            frame = self.DM_transform(frame)
        # series
        if self.sts_str:
            series = self.STS[:,idx-self.wdw_len*self.wdw_str+1:idx+1:self.wdw_str]
        else:
            series = self.STS[:,idx-self.wdw_len*self.wdw_str+1:idx+1]
        if self.STS_transform:
            series = self.STS_transform(series)
        # label
        label = self.SCS[idx]

        return {"frame": frame, "series": series, "label": label}

class StaticDM(LightningDataModule):

    """ Data module for the experiments. """

    STS: np.ndarray     # data stream
    SCS: np.ndarray     # class stream
    DM: np.ndarray      # dissimilarity matrix

    wdw_len: int        # window length
    wdw_str: int        # window stride
    sts_str: bool       # stride the series too?

    n_dims: int         # number of STS dimensions
    n_classes: int      # number of classes
    n_patterns: int     # number of patterns
    l_patterns: int     # pattern size
    
    data_split: dict[str: np.ndarray]    
                        # train / val / test split
    batch_size: int     # dataloader batch size
    
    random_seed: int    # random seed
    num_workers: int    # dataloader nworkers

    def __init__(self,
            STS: np.ndarray, SCS: np.ndarray, DM: np.ndarray,    
            wdw_len: int, wdw_str: int, sts_str: bool,
            data_split: dict, batch_size: int, 
            random_state: int = 42, num_workers: int = mp.cpu_count()//2
            ) -> None:
        
        # save parameters as attributes
        super(StaticDM).__init__(), self.__dict__.update(locals())

        # gather dataset info   
        self.n_dims = STS.shape[0]
        self.n_classes = len(np.unique(SCS))
        self.n_patterns = DM.shape[0]
        self.l_patterns = DM.shape[1]

        # convert to tensors
        self.STS = torch.from_numpy(STS).to(torch.float32)
        self.SCS = torch.from_numpy(SCS).to(torch.int8)
        self.DM = torch.from_numpy(DM).to(torch.float32)

        # generate datasets
        self.update(wdw_len=wdw_len, wdw_str=wdw_str, sts_str=sts_str,
            data_split=data_split, batch_size=batch_size)

    def update(self, wdw_len: int = None, wdw_str: int = None, sts_str: bool = None,
            data_split: dict = None, batch_size: int = None) -> None:

        """ Generate datasets based on the frame parameters. """

        # update values if provided
        for var, val in locals().items():
            if val is not None:
                self.__dict__[var] = val

        if data_split is not None:
            margin = self.wdw_len*self.wdw_str+1
            for split in data_split:
                if margin > len(data_split[split]):
                    data_split[split] = data_split[split][margin:]

        train_idx = self.data_split["train"]
        val_idx = self.data_split["val"]
        test_idx = self.data_split["test"]

        DM_trans = tv.transforms.Normalize(
            self.DM[:,:,train_idx].mean(axis=[1,2]),
            self.DM[:,:,train_idx].std(axis=[1,2]))
        
        ds_args = {"DM": self.DM, "STS": self.STS, "SCS": self.SCS,
        "wdw_len": self.wdw_len, "wdw_str": self.wdw_str, 
        "sts_str": self.sts_str, "DM_trans": DM_trans}

        self.ds_train = StaticDS(index=train_idx, **ds_args)
        self.ds_val = StaticDS(index=val_idx, **ds_args)
        self.ds_test = StaticDS(index=test_idx, **ds_args)

    def train_dataloader(self) -> DataLoader:
        """ Returns the training DataLoader. """
        return DataLoader(self.ds_train, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        """ Returns the validation DataLoader. """
        return DataLoader(self.ds_val, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)