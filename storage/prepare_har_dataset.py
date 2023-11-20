import pandas
import numpy as np
import os
import datetime

from label_mappings import *

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
            print(f"{len(event_length[c])} events in class {HARTH_LABELS[c]},")
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
    splits = [0] + list(np.nonzero(np.diff(subject).reshape(-1))[0] + 1) + [subject.size]

    save_path = os.path.join(dataset_dir, "UCI HAR Dataset", split)

    for split in range(len(splits) - 1):
        with open(os.path.join(save_path, f"subject_{int(subject[splits[split]])}_sensor.npy"), "wb") as f:
            np.save(f, STS[splits[split]:(splits[split+1])])
                    
        with open(os.path.join(save_path, f"subject_{int(subject[splits[split]])}_class.npy"), "wb") as f:
            np.save(f, SCS[splits[split]:(splits[split+1])])
            

if __name__ == "__main__":
    prepare_harth("storage/datasets/HARTH")