#!/usr/bin/env python

"""
Script that automates the experiments in a SLURM queue.
"""

from pathlib import Path
import subprocess

DATASETS = ["CBF"]                                  # Datasets             
ARCHS = ["ResNet"]                                  # Architectures
WINDOW_LENGTHS: list[int] = [10]                    # Window length
WINDOW_TIME_STRIDES: list[int] = [1, 3, 5, 7]       # Window time stride
WINDOW_PATT_STRIDES: list[int] = [2, 3, 5]          # Window pattern stride

RHO_DFS: float = 0.1                # Memory parameter for DF
BATCH_SIZE: bool = 128              # Batch size
VAL_SIZE: float = 0.25              # Validation size
STS_PRET_EVENTS = 1000              # Number of events for pretraining

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

account     = "bcam-exclusive"
partition   = "bcam-exclusive"
email       = "rcoterillo@bcamath.org"
env         = Path("/scratch/rcoterillo/s3ts/s3ts_env/bin/activate")
script      = Path("/scratch/rcoterillo/s3ts/pret_cli.py")
storage_dir = Path("/scratch/rcoterillo/s3ts/storage")

outputs = Path("outputs/").absolute()
outputs.mkdir(exist_ok=True)

logs = Path("logs/").absolute()
logs.mkdir(exist_ok=True)

jobs = Path("jobs/").absolute()
jobs.mkdir(exist_ok=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def launch_job(dataset: str, arch: str,
            window_length: int, 
            stride_series: bool,
            window_time_stride: int, 
            window_patt_stride: int   
            ) -> None: 

    ss = 1 if stride_series else 0
 
    job_name = f"{dataset}_wl{window_length}_ts{window_time_stride}_ps{window_patt_stride}_ss{ss}_{arch}"
    job_file = jobs / (job_name + ".job")
    log_file = logs / (job_name + ".log")
    out_file = outputs / (job_name + ".out")
    err_file = outputs / (job_name + ".err")
    gpu = False

    with job_file.open(mode="w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        f.write(f"#SBATCH --account={account}\n")
        f.write(f"#SBATCH --partition={account}\n")
        if gpu:
            f.write(f"#SBATCH --gres=gpu:rtx8000:1\n")
        f.write(f"#SBATCH --nodes=1\n")
        f.write(f"#SBATCH --ntasks-per-node=1\n")
        f.write(f"#SBATCH --cpus-per-task=6\n")
        f.write(f"#SBATCH --mem=10G\n")
        f.write(f"#SBATCH --mail-type=END\n")
        f.write(f"#SBATCH --mail-user={email}\n")
        f.write(f"#SBATCH -o {out_file}\n")
        f.write(f"#SBATCH -e {err_file}\n")
        f.write(f"module load CUDA/11.3.1\n")
        f.write(f"module load cuDNN/8.2.1.32-CUDA-11.3.1\n")
        f.write(f"module load Python/3.10.4-GCCcore-11.3.0\n")
        f.write(f"source {str(env)}\n")
        f.write(f"python {str(script)} " +\
                f"--dataset {dataset} " +\
                f"--arch {arch} " +\
                f"--stride_series {str(stride_series)} " +\
                f"--window_length {window_length} " +\
                f"--window_time_stride {window_time_stride} " +\
                f"--window_patt_stride {window_patt_stride} " +\
                f"--log_file {log_file} --storage_dir {str(storage_dir)}")
    subprocess.run(["sbatch", str(job_file)], capture_output=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for dataset in DATASETS:
    for arch in ARCHS:
        for wlen in WINDOW_LENGTHS:
            for wts in WINDOW_TIME_STRIDES:
                launch_job(dataset, arch, wlen, True, wts, 1)
                launch_job(dataset, arch, wlen, False, wts, 1)
        for wlen in WINDOW_LENGTHS:
            for wps in WINDOW_PATT_STRIDES:
                launch_job(dataset, arch, wlen, True, 7, wps)