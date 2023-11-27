# torch / lightning imports

import os

from torch.utils.data import Dataset, DataLoader
from s3ts.api.dms.base import StreamingFramesDM
from s3ts.api.encodings import compute_DM
import torchvision as tv
import torch

import sys

# standard library imports
import multiprocessing as mp
import numpy as np

from storage.har_datasets import STSDataset

class DFDataset(Dataset):
    def __init__(self, 
            stsds: STSDataset = None,
            patterns: np.ndarray = None,
            w: float = 0.1,
            dm_transform = None) -> None:
        super().__init__()

        '''
            patterns: shape (n_shapes, channels, pattern_size)
        '''

        self.stsds = stsds

        self.cache_dir = os.path.join(os.getcwd(), "df_cache")
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        elif len(os.listdir(self.cache_dir)) > 0:
            raise Exception(f"Cache directory not empty, please clean the directory {self.cache_dir}")

        self.patterns = patterns
        self.dm_transform = dm_transform

        self.rho = w

        for s in range(self.stsds.splits.shape[0] - 1):
            save_path = os.path.join(self.cache_dir, f"part{s}.npy")
            self._compute_dm_cache(patterns, self.stsds.splits[s:s+2], save_path)

    def _compute_dm_cache(self, pattern, split, save_path):
        DM = compute_DM(self.stsds.STS[:, split[0]:split[1]], pattern, rho=self.rho)

        with open(save_path, "wb") as f:
            np.save(f, DM)

    def __del__(self):
        print("called del")
        # clean cache
        if not self.cache_dir is None:
            [os.remove(os.path.join(self.cache_dir, f)) for f in os.listdir(self.cache_dir) if ".npy" in f]
            if len(os.listdir(self.cache_dir)) == 0:
                os.rmdir(self.cache_dir)

    def __len__(self):
        return len(self.stsds)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:

        id = self.stsds.indices[index]

        # identify the split of the index

        s = np.argwhere(self.stsds.splits > id)[0, 0] - 1
        first = id - self.stsds.wsize*self.stsds.wstride - self.stsds.splits[s]
        last = id - self.stsds.splits[s]

        temp = np.load(os.path.join(self.cache_dir, f"part{s}.npy"))[:, :, first:last:self.stsds.wstride] * 1 # *1 so that a new buffer is created

        dm = torch.from_numpy(temp).float()

        if not self.dm_transform is None:
            dm = self.dm_transform(dm)

        return (dm, 
                self.stsds.STS[:, first:last:self.stsds.wstride], 
                self.stsds.SCS[id])

class DFDatasetCopy(Dataset):
    def __init__(self,
            dfds: DFDataset, indices: np.ndarray) -> None:
        super().__init__()

        self.dfds = dfds
        self.indices = indices
        
    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, int]:

        df, ts, c = self.dfds[self.indices[index]]
        return {"frame": df, "series": ts, "label": c}
    
    def __del__(self):
        del self.dfds

class LDFDataset(StreamingFramesDM):

    """ Data module for the experiments. """

    STS: np.ndarray     # data stream
    SCS: np.ndarray     # class stream
    DM: np.ndarray      # dissimilarity matrix

    data_split: dict[str: np.ndarray]    
                        # train / val / test split
    batch_size: int     # dataloader batch size

    def __init__(self,
            dfds: DFDataset,    
            data_split: dict, batch_size: int, 
            random_seed: int = 42, 
            num_workers: int = mp.cpu_count()//2
            ) -> None:

        '''
            dfds: Dissimilarity frame DataSet
            data_split: How to split the dfds, example below 

            data_split = {
                "train" = lambda indices: train_condition,
                "val" = lambda indices: val_condition,
                "test" = lambda indices: test_condition
            } -> the train dataset will be indices from dfds.stsds.splits[0] to dfds.stsds.splits[0]
        '''

        # save parameters as attributes
        super().__init__()
        
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers

        self.dfds = dfds
        self.wdw_len = self.dfds.stsds.wsize
        self.wdw_str = self.dfds.stsds.wstride
        self.sts_str = False

        # gather dataset info   
        self.n_dims = self.dfds.stsds.STS.shape[1]
        self.n_classes = len(np.unique(self.dfds.stsds.SCS))
        self.n_patterns = self.dfds.patterns.shape[0]
        self.l_patterns = self.dfds.patterns.shape[2]

        # convert to tensors
        if not torch.is_tensor(self.dfds.stsds.STS):
            self.dfds.stsds.STS = torch.from_numpy(self.dfds.stsds.STS).to(torch.float32)
        if not torch.is_tensor(self.dfds.stsds.SCS):
            self.dfds.stsds.SCS = torch.from_numpy(self.dfds.stsds.SCS).to(torch.int64)

        train_indices = self.dfds.stsds.indices[data_split["train"](self.dfds.stsds.indices)]
        test_indices = self.dfds.stsds.indices[data_split["test"](self.dfds.stsds.indices)]
        val_indices = self.dfds.stsds.indices[data_split["val"](self.dfds.stsds.indices)]

        self.ds_train = DFDatasetCopy(self.dfds, train_indices)
        self.ds_test = DFDatasetCopy(self.dfds, test_indices)
        self.ds_val = DFDatasetCopy(self.dfds, val_indices)
        
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