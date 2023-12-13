import os
from time import time

start_time = time()
str_time = lambda b: f"{int(b//3600):02d}:{int((b%3600)//60):02d}:{int((b%3600)%60):02d}.{int(round(b%1, 3)*1000):03d}"

# dataset imports
from storage.har_datasets import *
from s3ts.api.dms.har_datasets import LDFDataset, DFDataset
from s3ts.dtw_layer.lightningstsds import LSTSDataset
from storage.label_mappings import *
from torchvision.transforms import Normalize

def load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize):
 
    if dataset_home_directory is None:
        dataset_home_directory = "./datasets"

    ds = None
    if dataset_name == "WISDM":
        ds = WISDMDataset(
            os.path.join(dataset_home_directory, dataset_name), 
            wsize=window_size, wstride=window_stride, normalize=normalize)
    elif dataset_name == "UCI-HAR":
        ds = UCI_HARDataset(
            os.path.join(dataset_home_directory, dataset_name), split="both", 
            wsize=window_size, wstride=window_stride, normalize=normalize, label_mapping=ucihar_label_mapping)
    elif dataset_name == "REALDISP":
        ds = REALDISPDataset(
            os.path.join(dataset_home_directory, dataset_name), sensor_position=["LLA", "BACK"], sensor=["ACC"], mode=["ideal"],
            wsize=window_size, wstride=window_stride, normalize=normalize)
    elif dataset_name == "HARTH":
        ds = HARTHDataset(
            os.path.join(dataset_home_directory, dataset_name), 
            wsize=window_size, wstride=window_stride, normalize=normalize, label_mapping=harth_label_mapping)
    elif dataset_name == "MHEALTH":
        ds = MHEALTHDataset(
            os.path.join(dataset_home_directory, dataset_name), sensor="acc",
            wsize=window_size, wstride=window_stride, normalize=normalize, label_mapping=harth_label_mapping)
    
    return ds

def load_dmdataset(
        dataset_name,
        dataset_home_directory = None,
        batch_size = 16,
        num_workers = 1,
        window_size = 32,
        window_stride = 1,
        normalize = True,
        pattern_size = None,
        compute_n = 500,
        subjects_for_test = None):
    
    pattern_size = window_size
    
    ds = load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize)
        
    print(f"Loaded dataset {dataset_name} with a total of {len(ds)} observations for window size {window_size}")

    # load medoids if already computed
    if not os.path.exists(os.path.join(dataset_home_directory, dataset_name, f"meds{window_size}.npz")):
        print("Computing medoids...")
        meds = sts_medoids(ds, n=compute_n)
        with open(os.path.join(dataset_home_directory, dataset_name, f"meds{window_size}.npz"), "wb") as f:
            np.save(f, meds)
    else:
        meds = np.load(os.path.join(dataset_home_directory, dataset_name, f"meds{window_size}.npz"))
        assert meds.shape[2] == pattern_size
    
    print("Computing/loading dissimilarity frames...")
    dfds = DFDataset(ds, patterns=meds, w=0.1, dm_transform=None, ram=False)

    data_split = split_by_test_subject(ds, subjects_for_test)

    if normalize:
        # get average values of the DM
        DM = []
        np.random.seed(42)
        for i in np.random.choice(np.arange(len(dfds))[data_split["train"](dfds.stsds.indices)], compute_n):
            dm, _, _ = dfds[i]
            DM.append(dm)
        DM = torch.stack(DM)

        dm_transform = Normalize(mean=DM.mean(dim=[0, 2, 3]), std=DM.std(dim=[0, 2, 3]))
        dfds.dm_transform = dm_transform

    dm = LDFDataset(dfds, data_split=data_split, batch_size=batch_size, random_seed=42, num_workers=num_workers)

    print(f"Using {len(dm.ds_train)} observations for training and {len(dm.ds_val)} observations for validation and test")

    return dm


def load_tsdataset(
        dataset_name,
        dataset_home_directory = None,
        batch_size = 16,
        num_workers = 1,
        window_size = 32,
        window_stride = 1,
        normalize = True,
        pattern_size = None,
        subjects_for_test = None):
    
    ds = load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize)
        
    print(f"Loaded dataset {dataset_name} with a total of {len(ds)} observations for window size {window_size}")

    data_split = split_by_test_subject(ds, subjects_for_test)

    dm = LSTSDataset(ds, data_split=data_split, batch_size=batch_size, random_seed=42, num_workers=num_workers)
    dm.l_patterns = pattern_size

    print(f"Using {len(dm.ds_train)} observations for training and {len(dm.ds_val)} observations for validation and test")

    return dm