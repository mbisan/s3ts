import pandas
import numpy as np
import os
import datetime

def prepare_harth(dataset_dir):

    ds = []

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
            # all .csv files have some timestamps with jumps of less or more than 20ms
            # for jumps with less than 10ms, we remove observations, for jumps with more than 20ms, we consider a new STS

            # remove observations closer than 20ms from each other
            remove = []
            j = 0
            for i in range(len(ds) - 1):
                if (ds["dt"][i + 1] - ds["dt"][j]) < 20:
                    remove.append(i + 1)
                else:
                    j = i

            ds.drop(remove, inplace=True)

            # extract STS from the data (segments with all the points in between at most 20ms apart)
            splits = []
            last = 0
            for i in range(len(ds) - 1):
                if (ds["dt"][i + 1] - ds["dt"][i]) > 20:
                    splits.append(ds.loc[last:i])
                    last = i + 1

            # print(np.count_nonzero(remove))
            if not os.path.exists(os.path.join(dataset_dir, f"{file.replace('.', '_')}")):
                os.mkdir(os.path.join(dataset_dir, f"{file.replace('.', '_')}"))

            for i, sp in enumerate(splits):
                with open(os.path.join(dataset_dir, f"{file.replace('.', '_')}/acc{i}.npy"), "wb") as f:
                    np.save(f, sp[["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]].to_numpy())
                
                with open(os.path.join(dataset_dir, f"{file.replace('.', '_')}/label{i}.npy"), "wb") as f:
                    np.save(f, sp[["label"]].to_numpy())

if __name__ == "__main__":
    prepare_harth("storage/datasets/HARTH")