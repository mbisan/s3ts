

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

class StreamingFramesDM(LightningDataModule):

    wdw_len: int        # window length
    wdw_str: int        # window stride
    sts_str: bool       # stride the series too?

    n_dims: int         # number of STS dimensions
    n_classes: int      # number of classes
    n_patterns: int     # number of patterns
    l_patterns: int     # pattern size

    batch_size: int     # dataloader batch size
    random_seed: int    # random seed
    num_workers: int    # dataloader nworkers

    ds_train: Dataset   # train dataset
    ds_val: Dataset     # validation dataset
    ds_test: Dataset    # test dataset