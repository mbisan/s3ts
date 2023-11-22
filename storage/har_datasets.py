import os
import wget

import zipfile
import gzip

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


if __name__ == "__main__":
    download("all", "./datasets")
    unpack("all", "./datasets")
