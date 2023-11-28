# %%
import numpy as np
import torch

# %%
from storage.har_datasets import *
from s3ts.api.dms.har_datasets import LDFDataset, DFDataset
from storage.label_mappings import *
from s3ts.api.nets.methods import create_model_from_DM, train_model, test_model

# %%
from torchvision.transforms import Normalize

# %%
UCI_HAR_LABELS

# %%
label_mapping = np.zeros(7)
label_mapping[1:] = np.arange(6)

ds = UCI_HARDataset("./datasets/UCI-HAR/", split="train", wsize=64, normalize=True, label_mapping=label_mapping)
meds = sts_medoids(ds, 1000)
dfds = DFDataset(ds, patterns=meds, w=0.1, dm_transform=None, ram=False)

# %%
DM = []

np.random.seed(42)
for i in np.random.choice(np.arange(len(dfds)), 5000):
    dm, _, _ = dfds[i]
    DM.append(dm)

DM = torch.stack(DM)

dm_transform = Normalize(mean=DM.mean(dim=[0, 2, 3]), std=DM.std(dim=[0, 2, 3]))

# %%
dfds.dm_transform = dm_transform

# %%
data_split = {
    "train": lambda x: x<300000,
    "val": lambda x: (x>=300000) * (x<375000),
    "test": lambda x: x>=375000
}

dm = LDFDataset(dfds, data_split=data_split, batch_size=32, random_seed=42, num_workers=8)

# %%
len(dm.ds_train) + len(dm.ds_val) + len(dm.ds_test)

# %%
model = create_model_from_DM(dm, name=None, 
        dsrc="img", arch="cnn", task="cls")

# %%
model, data = train_model(dm, model, max_epochs=2)
print(data)


