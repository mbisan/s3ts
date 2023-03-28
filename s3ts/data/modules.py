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
import logging as log
import numpy as np


class DFDataset(Dataset):

    """ Dataset for the experiments. """

    def __init__(self,
            DM: torch.Tensor,
            STS: torch.Tensor,
            SCS: torch.Tensor,
            index: np.ndarray,
            window_length: int,
            window_time_stride: int,
            window_pattern_stride: int,
            DM_transform = None,
            STS_transform = None, 
            SCS_transform = None,
            stride_series: bool = True,
            ) -> None:

        """ Initialize the dataset. 
        
        Parameters
        ----------
        DM : torch.Tensor
            Dissimilarity matrix
        STS : torch.Tensor
            Streaming Time Series
        SCS : torch.Tensor
            Streaming Class Series
        index : np.ndarray
            Index of the samples
        window_length : int
            Length of the window
        window_time_stride : int
            Time stride of the frame window
        window_pattern_stride : int
            Pattern stride of the frame window
        DM_transform : callable, optional
            Transformation to apply to the DM
        STS_transform : callable, optional
            Transformation to apply to the STS
        SCS_transform : callable, optional
            Transformation to apply to the SCS
        """

        super().__init__()

        self.DM = DM
        self.STS = STS
        self.SCS = SCS
        self.index = index

        self.window_length = window_length
        self.wts = window_time_stride
        self.wps = window_pattern_stride
        self.stride_series = stride_series

        self.available_events = 1

        self.DM_transform = DM_transform
        self.STS_transform = STS_transform
        self.SCS_transform = SCS_transform

    def __len__(self) -> int:

        """ Return the length of the dataset. """
        
        return len(self.index)


    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:

        """ Return an entry (x, y) from the dataset. """

        idx = self.index[idx]
       
        # Grab the frame
        frame = self.DM[:,:,idx - self.window_length*self.wts+1:idx+1:self.wts]
        if self.DM_transform:
            frame = self.DM_transform(frame)

        # Grab the series
        if self.stride_series:
            series = self.STS[idx - self.window_length*self.wts+1:idx+1:self.wts]
        else:
            series = self.STS[idx - self.window_length*self.wts+1:idx+1]
        if self.STS_transform:
            series = self.STS_transform(series)

        # Grab the label
        label = self.SCS[idx]
        if self.SCS_transform:
            label = self.SCS_transform(label)

        # Return the frame, series, and label
        return frame, series, label

# ================================================================= #

class DFDataModule(LightningDataModule):

    """ Data module for the experiments. """

    def __init__(self,
            STS_tra: np.ndarray, STS_pre: np.ndarray, 
            SCS_tra: np.ndarray, SCS_pre: np.ndarray,
            DM_tra: np.ndarray, DM_pre: np.ndarray,
            sample_length: int, patterns: np.ndarray, 
            val_size: float, batch_size: int, 
            STS_train_events: int, STS_pret_events: int,STS_test_events: int, 
            window_length: int, window_time_stride: int, window_pattern_stride: int, 
            stride_series: bool, pretrain: bool, random_state: int = 0, 
            num_workers: int = mp.cpu_count()//2
            ) -> None:
        
        """ Initialize the DFs data module.
        
        Parameters
        ----------
        STS_tra : np.ndarray
            Training Streaming Time Series
        STS_pre : np.ndarray
            Pre-training Streaming Time Series
        SCS_tra : np.ndarray
            Training Streaming Class Series
        SCS_pre : np.ndarray
            Pre-training Streaming Class Series
        DM_tra : np.ndarray
            Training Dissimilarity Matrix
        DM_pre : np.ndarray
            Pre-training Dissimilarity Matrix
        STS_train_events : int
            Number of events in the training STS
        STS_pret_events : int
            Number of events in the pretraining STS
        STS_test_events : int
            Number of events in the test STS
        event_length : int
            Length of the events
        patterns : np.ndarray
            Patterns for the DMs
        batch_size : int
            Batch size for the dataloaders
        window_length : int
            Length of the window for the frame
        window_time_stride : int
            Time stride of the frame window
        window_pattern_stride : int
            Pattern stride of the frame window
        stride_series : bool
            Whether to stride the series
        random_state : int, optional
            Random state for the data module
        num_workers : int, optional
            Number of workers for the dataloaders
        """
        
        super().__init__()

        # Register dataset parameters
        self.pretrain = pretrain
        self.val_size = val_size
        self.batch_size = batch_size
        self.window_length = window_length
        self.window_time_stride = window_time_stride
        self.window_pattern_stride = window_pattern_stride
        self.stride_series = stride_series
        self.random_state = random_state
        self.num_workers = num_workers

        # Gather dataset info
        self.sample_length = sample_length
        self.STS_train_events = STS_train_events
        self.STS_pret_events = STS_pret_events
        self.STS_test_events = STS_test_events
        self.n_classes = len(np.unique(SCS_tra))
        self.n_patterns = patterns.shape[0]
        self.l_patterns = patterns.shape[1]
    
        # Convert STS to tensors
        self.STS_tra = torch.from_numpy(STS_tra).to(torch.float32)
        self.STS_pre = torch.from_numpy(STS_pre).to(torch.float32)
        
        # Convert SCS to tensors
        self.SCS_tra = torch.from_numpy(SCS_tra).to(torch.int64)
        self.SCS_pre = torch.from_numpy(SCS_pre).to(torch.int64)
        self.labels_tra = torch.nn.functional.one_hot(self.SCS_tra, num_classes=self.n_classes)
        self.labels_pre = torch.nn.functional.one_hot(self.SCS_pre, num_classes=self.n_classes)

        # Convert DM to tensors
        self.DM_tra = torch.from_numpy(DM_tra).to(torch.float32)
        self.DM_pre = torch.from_numpy(DM_pre).to(torch.float32)

        # Convert patterns to tensors
        self.patterns = torch.from_numpy(patterns).to(torch.float32)

        # Create the indices
        self.create_sample_index()

        # Do some logging
        log.info(f"Events in training STS: {STS_train_events}")
        log.info(f"Frames in training STS: {len(self.train_indices)} ")
        self.tra_ratios = np.unique(SCS_tra[self.train_indices], return_counts=True)[1]/(STS_train_events*sample_length)
        log.info(f"Train STS class ratios: {self.tra_ratios}")

        log.info(f"Events in pretrain STS: {STS_pret_events}")
        log.info(f"Frames in pretrain STS: {len(self.pret_indices)}")
        self.pre_ratios = np.unique(SCS_pre[self.pret_indices], return_counts=True)[1]/(STS_pret_events*sample_length)
        log.info(f"Pretrain STS class ratios: {self.pre_ratios}")

        log.info(f"Events in testing STS: {STS_test_events}")
        log.info(f"Frames in testing STS: {len(self.test_indices)}")
        self.test_ratios = np.unique(SCS_pre[self.test_indices], return_counts=True)[1]/(STS_test_events*sample_length)
        log.info(f"Test STS class ratios: {self.test_ratios}")

        # Calculate the memory usage of the datasets
        self.DM_mem = self.DM_tra.element_size()*self.DM_tra.nelement() + self.DM_pre.element_size()*self.DM_pre.nelement()
        self.STS_mem = self.STS_tra.element_size()*self.STS_tra.nelement() + self.STS_pre.element_size()*self.STS_pre.nelement()
        self.SCS_mem = self.SCS_tra.element_size()*self.SCS_tra.nelement() + self.SCS_pre.element_size()*self.SCS_pre.nelement()

        log.info(f"DM  memory usage: {self.DM_mem/1e6} MB")
        log.info(f"STS memory usage: {self.STS_mem/1e6} MB")
        log.info(f"SCS memory usage: {self.SCS_mem/1e6} MB")

    def create_sample_index(self, 
            av_train_events: int = None,
            av_pret_events: int = None,
            av_test_events: int = None):
        
        """ Create the sample indeces for the datasets. """

        # Calculate the margin due to the window length
        margin = self.window_length*self.window_pattern_stride

        # Creathe the default indices
        self.train_indices = np.arange(margin, self.STS_train_events*self.sample_length)
        self.pret_indices = np.arange(margin, self.STS_pret_events*self.sample_length)
        self.test_indices = np.arange(self.STS_pret_events*self.sample_length + margin,
            (self.STS_pret_events + self.STS_test_events)*self.sample_length)
        
        # Set the available samples by default
        self.av_train_events = self.STS_train_events
        self.av_pret_events = self.STS_pret_events
        self.av_test_events = self.STS_test_events

        # Check requested available samples are not larger than the actual ones
        if av_train_events is not None:
            assert av_train_events <= self.STS_train_events, "Requested available training events are larger than the actual ones"
            self.train_indices = np.arange(margin, av_train_events*self.sample_length)
            self.av_train_events = av_train_events
        if av_pret_events is not None:
            assert av_pret_events <= self.STS_pret_events, "Requested available pretrain events are larger than the actual ones"
            self.pret_indices = np.arange(margin, av_pret_events*self.sample_length)
            self.av_pret_events = av_pret_events
        if av_test_events is not None:
            assert av_test_events <= self.STS_test_events, "Requested available test events are larger than the actual ones"
            self.test_indices = np.arange(self.STS_pret_events*self.sample_length + margin,
                (self.STS_pret_events + av_test_events)*self.sample_length)
            self.av_test_events = av_test_events

    def create_datasets(self):

        # Normalization transform for the frames
        DM_transform = tv.transforms.Normalize(
            self.DM_tra.mean(axis=[1,2]),
            self.DM_tra.std(axis=[1,2]))
        
        # Create the training datasets
        tra_tot_samples = len(self.train_indices)
        tra_train_samples = tra_tot_samples-int(tra_tot_samples*self.val_size)
        self.ds_tra_train = DFDataset(index=self.train_indices[:tra_train_samples],
            DM=self.DM_tra, STS=self.STS_tra, SCS=self.labels_tra,
            window_length=self.window_length, stride_series=self.stride_series, 
            window_time_stride=self.window_time_stride, window_pattern_stride=self.window_pattern_stride, 
            DM_transform=DM_transform)
        self.ds_tra_val   = DFDataset(index=self.train_indices[tra_train_samples:],
            DM=self.DM_tra, STS=self.STS_tra, SCS=self.labels_tra,
            window_length=self.window_length, stride_series=self.stride_series, 
            window_time_stride=self.window_time_stride, window_pattern_stride=self.window_pattern_stride, 
            DM_transform=DM_transform)

        # Create the pretraining datasets
        pre_tot_samples = len(self.pret_indices)
        pre_train_samples = pre_tot_samples-int(pre_tot_samples*self.val_size)
        self.ds_pre_train = DFDataset(index=self.pret_indices[:pre_train_samples],
            DM=self.DM_pre, STS=self.STS_pre, SCS=self.labels_pre,
            window_length=self.window_length, stride_series=self.stride_series,
            window_time_stride=self.window_time_stride, window_pattern_stride=self.window_pattern_stride,
            DM_transform=DM_transform)
        self.ds_pre_val   = DFDataset(index=self.pret_indices[pre_train_samples:],
            DM=self.DM_pre, STS=self.STS_pre, SCS=self.labels_pre,
            window_length=self.window_length, stride_series=self.stride_series,
            window_time_stride=self.window_time_stride, window_pattern_stride=self.window_pattern_stride,
            DM_transform=DM_transform)
        
        # Create the testing dataset
        self.ds_test = DFDataset(index=self.test_indices,
            DM=self.DM_pre, STS=self.STS_pre, SCS=self.labels_pre,
            window_length=self.window_length, stride_series=self.stride_series,
            window_time_stride=self.window_time_stride, window_pattern_stride=self.window_pattern_stride,
            DM_transform=DM_transform)

    def update_sample_availability(self, 
            av_train_events: int = None,
            av_pret_events: int = None,
            av_test_events: int = None):
        
        """ Updates the samples index based on the available events. """
        
        self.create_sample_index(av_train_events, av_pret_events, av_test_events)
        self.create_datasets()

    def train_dataloader(self):
        """ Returns the training DataLoader. """
        if self.pretrain:
            return DataLoader(self.ds_pre_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=False)
        return DataLoader(self.ds_tra_train, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        """ Returns the validation DataLoader. """
        if self.pretrain:
            return DataLoader(self.ds_pre_val, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=False)
        return DataLoader(self.ds_tra_val, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        """ Returns the predict DataLoader."""
        return self.test_dataloader()