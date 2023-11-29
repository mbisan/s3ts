import pandas
import numpy as np
import os
import datetime

import os
import wget

import zipfile

DATASETS = {
    "WARD": "https://people.eecs.berkeley.edu/~yang/software/WAR/WARD1.zip",
    "HASC": "http://bit.ly/i0ivEz",
    "UCI-HAR": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "WISDM": "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz", # tar.gz
    "USC-HAD": "https://sipi.usc.edu/had/USC-HAD.zip",
    "OPPORTUNITY": "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
    "UMAFall": "https://figshare.com/ndownloader/articles/4214283/versions/7",
    "UDC-HAR": "https://lbd.udc.es/research/real-life-HAR-dataset/data_raw.zip",
    "HARTH": "http://www.archive.ics.uci.edu/static/public/779/harth.zip",
    "UniMiB-SHAR": "https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip?dl=1",
    "REALDISP": "https://archive.ics.uci.edu/static/public/305/realdisp+activity+recognition+dataset.zip",
    "DAPHNET-FOG": "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip",
    "MHEALTH": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
    "TempestaTMD": "https://tempesta.cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz" # tar
}

def download(dataset_name = "all", dataset_dir=None):

    if dataset_name == "all":
        for name in DATASETS.keys():
            download(dataset_name=name, dataset_dir=dataset_dir)
        return None
    
    assert dataset_name in DATASETS.keys()

    if not os.path.exists(f"{dataset_dir}/{dataset_name}"):
        os.mkdir(f"{dataset_dir}/{dataset_name}")

    #check if directory is empty
    if not os.listdir(f"{dataset_dir}/{dataset_name}/"):
        # download zipped dataset
        print(f"Downloading dataset {dataset_name}")
        file = wget.download(DATASETS[dataset_name], out=f"{dataset_dir}/{dataset_name}/")
        print(f"{dataset_name} downloaded to {file}")

def unpack(dataset_name = "all", dataset_dir=None):

    if dataset_name == "all":
        for name in DATASETS.keys():
            unpack(dataset_name=name, dataset_dir=dataset_dir)
        return None

    if not os.path.exists(f"{dataset_dir}/{dataset_name}") or not os.listdir(f"{dataset_dir}/{dataset_name}/"):
        return print(f"Dataset is not downloaded to {dataset_dir}/{dataset_name}/")
    
    assert dataset_name in DATASETS.keys()
    assert os.path.exists(f"{dataset_dir}/{dataset_name}")

    files = os.listdir(f"{dataset_dir}/{dataset_name}")
    assert len(files) > 0

    if len(files) > 1:
        return print(f"Dataset {dataset_name} already unpacked")

    if dataset_name in ["WISDM", "TempestaTMD"]:
        # use gzip to decompress
        pass
    else:
        print(os.path.join(f"{dataset_dir}/{dataset_name}", files[-1]))
        # use zipfile to decompress
        with zipfile.ZipFile(os.path.join(f"{dataset_dir}/{dataset_name}", files[-1]), "r") as zip_ref:
            zip_ref.extractall(f"{dataset_dir}/{dataset_name}")

        # unpack zip files inside zip file
        files_new = os.listdir(f"{dataset_dir}/{dataset_name}")
        for new_file in files_new:
            if new_file[-3:] == "zip" and new_file not in files:
                with zipfile.ZipFile(os.path.join(f"{dataset_dir}/{dataset_name}", new_file), "r") as zip_ref:
                    zip_ref.extractall(f"{dataset_dir}/{dataset_name}")

############################################################################################

def prepare_harth(dataset_dir):

    ds = []

    counts = {}
    event_length = {}

    for dir, _, files in os.walk(os.path.join(dataset_dir, "harth")):
        for i, file in enumerate(files):
            print(file)
            ds = pandas.read_csv(os.path.join(dir, file))

            # change timestamp to time (from 0)
            if len(ds["timestamp"][0]) == 29: # one of the .csv files has incorrect formatting on milliseconds
                ds["dt"] = list(map(lambda x : (datetime.datetime.strptime(x[:-3], "%Y-%m-%d %H:%M:%S.%f") - \
                                datetime.datetime.strptime(ds["timestamp"][0][:-3], "%Y-%m-%d %H:%M:%S.%f")) / datetime.timedelta(milliseconds=1), ds["timestamp"]) )
            else:
                ds["dt"] = list(map(lambda x : (datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") - \
                            datetime.datetime.strptime(ds["timestamp"][0], "%Y-%m-%d %H:%M:%S.%f")) / datetime.timedelta(milliseconds=1), ds["timestamp"]) )

            # check if the sampling rate is correct, at 50hz
            # all .csv files have some timestamps with jumps of less or more than 15ms
            # for jumps with less than 10ms, we remove observations, for jumps with more than 25ms, we consider a new STS

            remove = []
            j = 0
            for i in range(len(ds) - 1):
                diff = ds["dt"][i + 1] - ds["dt"][j]
                if diff < 15:
                    remove.append(i + 1)
                else:
                    j = i + 1

            print(f"Removed {len(remove)} observations.")
            ds.drop(remove, inplace=True)
            ds = ds.reset_index()

            splits = []
            last = 0
            for i in range(len(ds) - 1):
                if (ds["dt"][i + 1] - ds["dt"][i]) > 25:
                    splits.append(ds.loc[last:i])
                    last = i + 1
            splits.append(ds.loc[last:len(ds)])

            if not os.path.exists(os.path.join(dataset_dir, f"{file.replace('.csv', '')}")):
                os.mkdir(os.path.join(dataset_dir, f"{file.replace('.csv', '')}"))

            for i, sp in enumerate(splits):
                labels = sp[["label"]].to_numpy()

                with open(os.path.join(dataset_dir, f"{file.replace('.csv', '')}/acc{i}.npy"), "wb") as f:
                    np.save(f, sp[["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]].to_numpy())
                
                with open(os.path.join(dataset_dir, f"{file.replace('.csv', '')}/label{i}.npy"), "wb") as f:
                    np.save(f, labels)
                
                # update class counts
                lb, c = np.unique(labels, return_counts=True)
                for i, l in enumerate(lb):
                    counts[l] = counts.get(l, 0) + c[i]

                # update event counts and event length
                current_event = 0
                for i in range(1, labels.size - 1):
                    if labels[i] != labels[current_event]:
                        event_length[int(labels[current_event])] = \
                            event_length.get(int(labels[current_event]), []) + [i - current_event]
                        current_event = i
                
                # last event
                event_length[int(labels[current_event])] = \
                            event_length.get(int(labels[current_event]), []) + [labels.size - current_event]
    
    # print statistics
    total = sum(counts.values())
    print(f"Total number of observations: {total}")

    for c in counts.keys():
            print(f"{len(event_length[c])} events in class {c},")
            print(f"\twith size (min) {min(event_length[c])}, (max) {max(event_length[c])}, (mean) {np.mean(event_length[c])}")
            print(f"\t{counts[c]} observations ({(counts[c]/total):.2f})")


def prepare_uci_har(dataset_dir, split = "both"):
    # Dataset is already split into windows (128-point windows, or 2.56s at 20ms per observation), with 50% (64-points) overlap
    # recover the STS, we assume no data points are missing

    assert split in ["both", "train", "test"]

    if split == "both":
        prepare_uci_har(dataset_dir, split="train")
        prepare_uci_har(dataset_dir, split="test")
        return 0
    
    labels = np.loadtxt(os.path.join(dataset_dir, "UCI HAR Dataset", split, f"y_{split}.txt"))

    # load data
    files = os.listdir(os.path.join(dataset_dir, "UCI HAR Dataset", split, "Inertial Signals"))
    data = {}

    for file in files:
        data[file] = []
        with open(os.path.join(dataset_dir, "UCI HAR Dataset", split, "Inertial Signals", file)) as f:
            for line in f:
                data[file].append(list(map(lambda x: float(x), line.strip().split())))
        data[file] = np.array(data[file])

    obs, ws = data[files[0]].shape
    overlap = ws//2
    total_points = (obs + 1) * overlap

    files.sort()
    # OPTIONAL remove total acceleration
    files = files[:-3]

    # recover STS and SCS
    STS = np.empty((total_points, len(files)))
    SCS = np.empty(total_points)

    for j, file in enumerate(files):
        for i in range(obs):
            STS[i*overlap:(i+1)*overlap, j] = data[file][i,:overlap]
            SCS[i*overlap:(i+1)*overlap] = labels[i]
        STS[obs*overlap:, j] = data[file][obs - 1,overlap:]
        SCS[obs*overlap:] = labels[obs - 1]

    # split into subjects
    subject = np.loadtxt(os.path.join(dataset_dir, "UCI HAR Dataset", split, f"subject_{split}.txt"))
    splits = [0] + list(np.nonzero(np.diff(subject).reshape(-1))[0] + 1) + [subject.size+1]

    save_path = os.path.join(dataset_dir, "UCI HAR Dataset", split)

    for split in range(len(splits) - 1):
        with open(os.path.join(save_path, f"subject_{int(subject[splits[split]])}_sensor.npy"), "wb") as f:
            np.save(f, STS[splits[split]*overlap:(splits[split+1])*overlap])
                    
        with open(os.path.join(save_path, f"subject_{int(subject[splits[split]])}_class.npy"), "wb") as f:
            np.save(f, SCS[splits[split]*overlap:(splits[split+1])*overlap])
            

def prepare_mhealth(dataset_dir):
    files = os.listdir(os.path.join(dataset_dir, "MHEALTHDATASET")) # contains .log files
    # Filter the files
    files = [f for f in files if ".log" in f]
    assert len(files) == 10 # dataset has 10 subjects

    # Assume no missing data
    # extract features
    columns = [
        "chest_acc_x", "chest_acc_y", "chest_acc_z",
        "ecg_1", "ecg_2",
        "lankle_acc_x", "lankle_acc_y", "lankle_acc_z",
        "lankle_gyro_x", "lankle_gyro_y", "lankle_gyro_z",
        "lankle_mag_x", "lankle_mag_y", "lankle_mag_z", 
        "rankle_acc_x", "rankle_acc_y", "rankle_acc_z",
        "rankle_gyro_x", "rankle_gyro_y", "rankle_gyro_z",
        "rankle_mag_x", "rankle_mag_y", "rankle_mag_z",
        "label" 
    ]

    for i, f in enumerate(files):
        ds = pandas.read_csv(os.path.join(dataset_dir, "MHEALTHDATASET", f), header=None, sep="\t")
        ds.columns = columns

        # save acc and other sensor data separately
        with open(os.path.join(dataset_dir, f"acc_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["chest_acc_x", "chest_acc_y", "chest_acc_z",
                                  "lankle_acc_x", "lankle_acc_y", "lankle_acc_z",
                                  "rankle_acc_x", "rankle_acc_y", "rankle_acc_z"]].to_numpy())
        
        with open(os.path.join(dataset_dir, f"mag_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["lankle_mag_x", "lankle_mag_y", "lankle_mag_z",
                                  "rankle_mag_x", "rankle_mag_y", "rankle_mag_z"]].to_numpy())
            
        with open(os.path.join(dataset_dir, f"gyro_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["lankle_gyro_x", "lankle_gyro_y", "lankle_gyro_z",
                                  "rankle_gyro_x", "rankle_gyro_y", "rankle_gyro_z"]].to_numpy())

        with open(os.path.join(dataset_dir, f"ecg_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["ecg_1", "ecg_2"]].to_numpy())

        with open(os.path.join(dataset_dir, f"labels_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["label"]].to_numpy())    

if __name__ == "__main__":
    # download("all", "./datasets")
    # unpack("all", "./datasets")

    # prepare_uci_har("./datasets/UCI-HAR")
    # prepare_harth("./datasets/HARTH")
    prepare_mhealth("./datasets/MHEALTH")