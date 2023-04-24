#!/usr/bin/env python

"""
Script that automates the experiments in a SLURM queue.
"""

from s3ts.hooks import launch_pret_sbatch
from s3ts.presets import BCAM_HIPATIA

ARCHS = {
    "DF": ["CNN", "ResNet"],
    "TS": ["RNN", "CNN", "ResNet"]
}
DATASETS = [
    "CBF"
]
   
# Parameter to study  
WINDOW_LENGTHS: list[int] = [10]                    # Window length (Section 4.1)
WINDOW_TIME_STRIDES: list[int] = [1, 3, 5, 7]       # Window time stride (Section ...)
WINDOW_PATT_STRIDES: list[int] = [2, 3, 5]          # Window pattern stride
EVENTS_PER_CLASS: list[int] = [8, 16, 24, 32]       # Number of events per class

# Fixed parameters
RHO_DFS: float = 0.1                # Memory parameter for DF
BATCH_SIZE: bool = 128              # Batch size
VAL_SIZE: float = 0.25              # Validation size
STS_LENGTH      = 1000              # Number of events for pretraining
MAX_EPOCH       = 60                # Number of epochs for pretraining 
LEARNING_RATE   = 1e-04             # Learning rate
RANDOM_STATE    = 0                 # Random state
NREPS = 5                           # Number of CV repetitions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


for repr in ARCHS:
    for arch in ARCHS[repr]:
        for dataset in DATASETS:
            for fold in range(NREPS):
            
                job_name = f"f{fold}_{repr}_{dataset}_{arch}_{EXP}"
                job_file = jobs / (job_name + ".job")
                log_file = logs / (job_name + ".log")
                out_file = outputs / (job_name + ".out")
                err_file = outputs / (job_name + ".err")
                
                gpu = (repr == "DF")
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
                            f"--repr {repr} --arch {arch} --exp {EXP} " + \
                            f"--log_file {str(log_file)}  --rep {fold} " + \
                            f"--cache_dir {str(cache_dir)}")
                subprocess.run(["sbatch", str(job_file)], capture_output=True)



    #             for dataset in DATASETS:
    # for arch in ARCHS:
    #     for wlen in WINDOW_LENGTHS:
    #         for wts in WINDOW_TIME_STRIDES:

    #             launch_pret_sbatch(
    #                 dataset=dataset, arch=arch, 
    #                 sts_length=STS_LENGTH, rho_dfs=RHO_DFS,
    #                 batch_size=BATCH_SIZE, val_size=VAL_SIZE,
    #                 max_epoch=MAX_EPOCH, learning_rate=LEARNING_RATE,
    #                 window_length=wlen, stride_series=True,
    #                 window_time_stride=wts, window_patt_stride=1,
    #                 random_state=RANDOM_STATE, **BCAM_HIPATIA)
                
    #             launch_pret_sbatch(
    #                 dataset=dataset, arch=arch, 
    #                 sts_length=STS_LENGTH, rho_dfs=RHO_DFS,
    #                 batch_size=BATCH_SIZE, val_size=VAL_SIZE,
    #                 max_epoch=MAX_EPOCH, learning_rate=LEARNING_RATE,
    #                 window_length=wlen, stride_series=False,
    #                 window_time_stride=wts, window_patt_stride=1,
    #                 random_state=RANDOM_STATE, **BCAM_HIPATIA)
                        
    #         for wps in WINDOW_PATT_STRIDES:

    #             launch_pret_sbatch(
    #                 dataset=dataset, arch=arch, 
    #                 sts_length=STS_LENGTH, rho_dfs=RHO_DFS,
    #                 batch_size=BATCH_SIZE, val_size=VAL_SIZE,
    #                 max_epoch=MAX_EPOCH, learning_rate=LEARNING_RATE,
    #                 window_length=wlen, stride_series=False,
    #                 window_time_stride=7, window_patt_stride=wps,
    #                 random_state=RANDOM_STATE, **BCAM_HIPATIA)