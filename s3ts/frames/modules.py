# external imports
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision as tv
import torch

import multiprocessing as mp
import numpy as np

# ================================================================= #

class FramesDataset(Dataset):

    def __init__(self,
            frames: np.ndarray,
            series: np.ndarray,
            labels: np.ndarray,
            indexes: np.ndarray,
            lab_shifts: list[int],
            window_size: int,
            transform = None, 
            target_transform = None
            ) -> None:

        self.frames = frames
        self.series = series
        self.labels = labels
        self.indexes = indexes

        self.n_shifts = len(lab_shifts)
        self.n_samples = len(self.indexes)
        self.window_size = window_size
        self.lab_shifts = lab_shifts

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """ Return the number of samples in the dataset. """
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:

        """ Return an entry (x, y) from the dataset. """

        idx = self.indexes[idx]
        
        if self.n_shifts > 1:
            label = []
            for s in self.lab_shifts:
                label.append(self.labels[idx+s,:])
            label = torch.stack(label)
        else:
            label = self.labels[idx+self.lab_shifts[0],:]

        frame = self.frames[:,:,idx - self.window_size:idx]
        
        if self.transform:
            frame = self.transform(frame)
        if self.target_transform:
            label = self.target_transform(label)

        return frame, label

# ================================================================= #

class FramesDataModule(LightningDataModule):

    def __init__(self,
            # calculate this outside
            STS_train: np.ndarray,
            DFS_train: np.ndarray,
            labels_train: np.ndarray,
            nframes_train: int, 
            # ~~~~~~~~~~~~~~~~~~~~~~
            window_size: int, 
            batch_size: int,
            lab_shifts: list[int],
            random_state: int = 0,
            # ~~~~~~~~~~~~~~~~~~~~~~
            STS_test: np.ndarray = None,
            DFS_test: np.ndarray = None,
            labels_test: np.ndarray = None,
            nframes_test: int = None,
            ) -> None:
        
        super().__init__()

        # convert dataset to tensors
        self.STS_train = torch.from_numpy(STS_train).to(torch.float32)
        self.DFS_train = torch.from_numpy(DFS_train).to(torch.float32)
        self.labels_train = torch.from_numpy(labels_train).to(torch.int64)
        self.labels_train = torch.nn.functional.one_hot(self.labels_train)

        # get dataset info
        self.n_labels = len(np.unique(labels_train))
        self.n_patterns = DFS_train.shape[0]
        self.l_patterns = DFS_train.shape[1]
        self.l_DFS_train = DFS_train.shape[2]

        # datamodule settings
        self.batch_size = batch_size
        self.window_size = window_size
        self.frame_buffer = 3*window_size
        self.lab_shifts = np.array(lab_shifts, dtype=int)
        self.random_state = random_state 
        self.num_workers = mp.cpu_count()//2        

        train_stop = int(nframes_train*2./3.)
        self.train_idx = np.arange(self.frame_buffer, self.frame_buffer+train_stop)
        self.valid_idx = np.arange(self.frame_buffer+train_stop, self.l_DFS_train-np.max(lab_shifts))
        print("Train samples:", len(self.train_idx))
        print("Valid samples:", len(self.valid_idx))

        # normalization_transform
        transform = tv.transforms.Normalize(
                self.DFS_train.mean(axis=[1,2]),
                self.DFS_train.std(axis=[1,2]))

        # training dataset
        self.ds_train = FramesDataset(
            indexes=self.train_idx, lab_shifts=lab_shifts,
            frames=self.DFS_train, series=self.STS_train, labels=self.labels_train,
            window_size=self.window_size, transform=transform)

        # validation dataset
        self.ds_eval = FramesDataset(
            indexes=self.valid_idx, lab_shifts=lab_shifts,
            frames=self.DFS_train, series=self.STS_train, labels=self.labels_train, 
            window_size=self.window_size, transform=transform)

        # repeat for testing if needed
        self.test = STS_test is not None
        if self.test:
            self.STS_test = torch.from_numpy(STS_test).to(torch.float32)
            self.DFS_test = torch.from_numpy(DFS_test).to(torch.float32)
            self.labels_test = torch.from_numpy(labels_test).to(torch.int64)
            self.labels_test = torch.nn.functional.one_hot(self.labels_test)
            self.l_DFS_test = DFS_test.shape[2]
            self.test_idx = np.arange(self.frame_buffer, nframes_test-np.max(lab_shifts))
            print("Test samples:", len(self.test_idx))

            self.ds_test = FramesDataset(
                indexes=self.test_idx, lab_shifts=lab_shifts,
                frames=self.DFS_test, series=self.STS_test, labels=self.labels_test, 
                window_size=self.window_size, transform=transform)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def train_dataloader(self):
        """ Returns the training DataLoader. """
        return DataLoader(self.ds_train, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def val_dataloader(self):
        """ Returns the validation DataLoader. """
        return DataLoader(self.ds_eval, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def test_dataloader(self):
        """ Returns the test DataLoader. """
        if self.test:
            return DataLoader(self.ds_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=False)
        return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def predict_dataloader(self):
        """ Returns the pred DataLoader. (test) """
        if self.test:
            return DataLoader(self.ds_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=False)
        return None