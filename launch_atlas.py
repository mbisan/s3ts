#!/usr/bin/env python

"""
Script that automates the experiments in a SLURM queue.
"""

from pathlib import Path
import subprocess

EXP = "stride"
ARCHS = {
    "DF": {"CNN", "ResNet"},
    #"TS": ("RNN", "CNN", "ResNet")
}
DATASETS = ["GunPoint", "Coffee", "PowerCons", "Plane", "CBF"]
N_SPLITS = 5

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

account = "bcam-exclusive"
partition = "bcam-exclusive"
email = "rcoterillo@bcamath.org"
env = Path("/scratch/rcoterillo/s3ts/s3ts_env/bin/activate")
script = Path("/scratch/rcoterillo/s3ts/main_cli.py")

outputs = Path("outputs/").absolute()
outputs.mkdir(exist_ok=True)

logs = Path("logs/").absolute()
logs.mkdir(exist_ok=True)

jobs = Path("jobs/").absolute()
jobs.mkdir(exist_ok=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for mode in ARCHS:
    for arch in ARCHS[mode]:
        for dataset in DATASETS:
            for fold in range(N_SPLITS):
            
                job_name = f"{arch}_{mode}_{EXP}_{dataset}_f{fold}"
                job_file = jobs / (job_name + ".job")
                log_file = logs / (job_name + ".log")
                out_file = outputs / (job_name + ".out")
                err_file = outputs / (job_name + ".err")
                
                gpu = (mode == "DF")
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
                    f.write(f"#SBATCH --cpus-per-task=12\n")
                    f.write(f"#SBATCH --mem=24G\n")
                    f.write(f"#SBATCH --mail-type=END\n")
                    f.write(f"#SBATCH --mail-user={email}\n")
                    f.write(f"#SBATCH -o {out_file}\n")
                    f.write(f"#SBATCH -e {err_file}\n")
                    f.write(f"module load CUDA/11.3.1\n")
                    f.write(f"module load cuDNN/8.2.1.32-CUDA-11.3.1\n")
                    f.write(f"module load Python/3.10.4-GCCcore-11.3.0\n")
                    f.write(f"source {str(env)}\n")
                    f.write(f"python {str(script)} --dataset {dataset} " + \
                            f"--mode {mode} --arch {arch} --exp {EXP} " + \
                            f"--log_file {str(log_file)} --fold {fold}")
                subprocess.run(["sbatch", str(job_file)], capture_output=True)