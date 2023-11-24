# torch / lightning imports
from torch.utils.data import Dataset, DataLoader
from s3ts.api.dm.base import StreamingFramesDM
from s3ts.api.encodings import compute_DM
import torchvision as tv
import torch

# standard library imports
import multiprocessing as mp
import numpy as np