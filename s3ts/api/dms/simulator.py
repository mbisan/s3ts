#/usr/bin/env python3
# -*- coding: utf-8 -*-

# torch / lightning imports

from torch.utils.data import Dataset, DataLoader
from s3ts.api.dm.base import StreamingFramesDM


import torchvision as tv
import torch

from s3ts.api.simulation import StreamSimulator


# standard library imports
import multiprocessing as mp
import numpy as np



class DynamicDS(Dataset):

    DM: torch.Tensor
    STS: torch.Tensor
    SCS: torch.Tensor
    wdw_len: int
    wdw_str: int
    sts_str: bool
    DM_trans: torch.nn.Module
    STS_trans: torch.nn.Module

    def __init__(self, DM, STS, SCS, wdw_len, wdw_str, sts_str,
            index=None, DM_trans=None, STS_trans=None) -> None:
        
        # save parameters as attributes
        super().__init__(), self.__dict__.update(locals())

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        # index
        idx = self.index[idx]
        # frame
        frame = self.DM[:,::self.wdw_str,idx - self.wdw_len*self.wdw_str+1:idx+1:self.wdw_str]
        if self.DM_trans:
            frame = self.DM_trans(frame)
        # series
        if self.sts_str:
            series = self.STS[:,idx-self.wdw_len*self.wdw_str+1:idx+1:self.wdw_str]
        else:
            series = self.STS[:,idx-self.wdw_len*self.wdw_str+1:idx+1]
        if self.STS_trans:
            series = self.STS_trans(series)
        # label
        label = self.SCS[idx]

        return {"frame": frame, "series": series, "label": label}

class DynamicDM(StreamingFramesDM):

    """ Data module for the experiments. """

    X: np.ndarray       # data stream
    Y: np.ndarray       # class stream
    DM: np.ndarray      # dissimilarity matrix

    data_split: dict[str: np.ndarray]    
                        # train / val / test split
    nsamps: dict[str: np.ndarray]    
                        # train / val / test nsamps
    pskip: float        # probability to skip a frame
    batch_size: int     # dataloader batch size
    
    random_seed: int    # random seed
    num_workers: int    # dataloader nworkers

    def __init__(self, X: np.ndarray, Y: np.ndarray, 
            patts: np.ndarray, img_type: str,    
            wdw_len: int, wdw_str: int, sts_str: bool,
            batch_size: int, data_split: dict, 
            nsamps: dict, pskip: float,
            random_state: int = 42, 
            num_workers: int = mp.cpu_count()//2
            ) -> None:
        
        # save parameters as attributes
        super().__init__(), self.__dict__.update(locals())

        # gather dataset info   
        self.n_dims = X.shape[1]
        self.n_classes = len(np.unique(Y))
        self.n_patterns = patts.shape[0]
        self.l_patterns = patts.shape[2]

        train_idx = self.data_split["train"]
        val_idx = self.data_split["val"]
        test_idx = self.data_split["test"]

        for stage in ["train", "val", "test"]:

            sim = StreamSimulator(X= X[data_split[stage]],
                Y = X[data_split[stage]], patts=patts,
                wdw_len=wdw_len, wdw_str=wdw_str,
                infinite_STS=)
            
        
        # convert to tensors
        self.STS = torch.from_numpy(STS).to(torch.float32)
        self.SCS = torch.from_numpy(SCS).to(torch.int64)
        self.DM = torch.from_numpy(DM).to(torch.float32)

        if data_split is not None:
            margin = self.wdw_len*self.wdw_str+1
            for split in data_split:
                if margin < len(data_split[split]):
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
    
    def predict_dataloader(self) -> DataLoader:
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)
    