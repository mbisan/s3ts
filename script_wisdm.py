import os

from time import time

start_time = time()

str_time = lambda b: f"{int(b//3600):02d}:{int((b%3600)//60):02d}:{int((b%3600)%60):02d}.{int(round(b%1, 3)*1000):03d}" if b//(3600*24) < 1 else \
                     f"{int(b//86400)}-{int((b%86400)//3600):02d}:{int(((b%86400)%3600)//60):02d}:{int(((b%86400)%3600)%60):02d}.{int(round(b%1, 3)*1000):03d}"


import numpy as np
import torch

from storage.har_datasets import WISDMDataset, sts_medoids, split_by_test_subject
from s3ts.api.dms.har_datasets import LDFDataset, DFDataset
from storage.label_mappings import *
from s3ts.api.nets.methods import create_model_from_DM, train_model, test_model
from torchvision.transforms import Normalize

# this dataset comes at 20hz
ds = WISDMDataset("./datasets/WISDM/", wsize=32, normalize=True, label_mapping=None)
print(len(ds))

if not os.path.exists("./datasets/WISDM/meds.npz"):
    print("Computing medoids")
    meds = sts_medoids(ds, n=500)
    with open("./datasets/WISDM/meds.npz", "wb") as f:
        np.save(f, meds)
else:
    meds = np.load("./datasets/WISDM/meds.npz")

print("Loading DFs")
dfds = DFDataset(ds, patterns=meds, w=0.1, dm_transform=None, ram=False)

DM = []

np.random.seed(42)
for i in np.random.choice(np.arange(len(dfds)), 500):
    dm, _, _ = dfds[i]
    DM.append(dm)

DM = torch.stack(DM)

dm_transform = Normalize(mean=DM.mean(dim=[0, 2, 3]), std=DM.std(dim=[0, 2, 3]))
dfds.dm_transform = dm_transform

data_split = split_by_test_subject(ds, 35)

dm = LDFDataset(dfds, data_split=data_split, batch_size=128, random_seed=42, num_workers=8)

print(f"Total points in the dataset: {len(dm.ds_train) + len(dm.ds_val) + len(dm.ds_test)}")
print("Train:", len(dm.ds_train))
print("Val:", len(dm.ds_val))
print("Test:", len(dm.ds_test))

model = create_model_from_DM(dm, name=None, 
        dsrc="img", arch="cnn", task="cls")

model, data = train_model(dm, model, max_epochs=10)
print(data)

print("Elapsed time:", str_time(time()-start_time))
