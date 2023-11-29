import numpy as np
import torch

from time import time

start_time = time()

str_time = lambda b: f"{int(b//3600):02d}:{int((b%3600)//60):02d}:{int((b%3600)%60):02d}.{int(round(b%1, 3)*1000):03d}" if b//(3600*24) < 1 else \
                     f"{int(b//86400)}-{int((b%86400)//3600):02d}:{int(((b%86400)%3600)//60):02d}:{int(((b%86400)%3600)%60):02d}.{int(round(b%1, 3)*1000):03d}"

from storage.har_datasets import *
from s3ts.api.dms.har_datasets import LDFDataset, DFDataset
from storage.label_mappings import *
from s3ts.api.nets.methods import create_model_from_DM, train_model, test_model

from torchvision.transforms import Normalize

UCI_HAR_LABELS

label_mapping = np.zeros(7)
label_mapping[1:] = np.arange(6)

ds = UCI_HARDataset("./datasets/UCI-HAR/", split="train", wsize=64, normalize=True, label_mapping=label_mapping)

print("Computing medoids")
a = time()
meds = sts_medoids(ds, 500)
print(f"Finished in {str_time(time()-a)}")

print("Computing/loading DF")
a = time()
dfds = DFDataset(ds, patterns=meds, w=0.1, dm_transform=None, ram=False)
print(f"Finished in {str_time(time()-a)}")

DM = []

np.random.seed(42)
for i in np.random.choice(np.arange(len(dfds)), 5000):
    dm, _, _ = dfds[i]
    DM.append(dm)

DM = torch.stack(DM)

dm_transform = Normalize(mean=DM.mean(dim=[0, 2, 3]), std=DM.std(dim=[0, 2, 3]))

dfds.dm_transform = dm_transform

data_split = {
    "train": lambda x: x<300000,
    "val": lambda x: (x>=300000) * (x<375000),
    "test": lambda x: x>=375000
}

dm = LDFDataset(dfds, data_split=data_split, batch_size=32, random_seed=42, num_workers=16)

print(f"Total points in the dataset: {len(dm.ds_train) + len(dm.ds_val) + len(dm.ds_test)}")

model = create_model_from_DM(dm, name=None, 
        dsrc="img", arch="cnn", task="cls", enc_feats=64)

print("Training start")

model, data = train_model(dm, model, max_epochs=20)
print(data)

print(str_time(time()-start_time))
