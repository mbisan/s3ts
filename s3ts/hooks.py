
from pathlib import Path
import subprocess

def launch_sbatch(
        dataset: str,               # Dataset name 
        repr: str,                  # Representation name
        arch: str,                  # Architecture name
        pret_sts_length: int,       # Length of the time series
        rho_dfs: float,             # Memory parameter for the DFs
        batch_size: int,            # Batch size
        val_size: float,            # Validation size
        max_epoch: int,             # Maximum number of epochs
        learning_rate: float,       # Learning rate
        window_length: int,         # Length of the window
        stride_series: bool,        # Whether to stride the series or not
        window_time_stride: int,    # Window time stride
        window_patt_stride: int,    # Window pattern stride
        random_state: int,          # Random state
        pretrain: bool,             # Whether to pretrain or not
        # ~~~~~~~~~~~~~ HPC Profile Settings ~~~~~~~~~~~~~
        job_name: str,          # Name of the job
        email: str,             # Email to send notifications
        cpu: int,               # Number of CPUs 
        mem: int,               # GB of RAM memory
        account: str,           # Account to charge
        partition: str,         # Partition to use
        venv_path: Path,        # Path to the virtual environment
        train_cli_script: Path, # Path to the CLI script for training
        pret_cli_script: Path,  # Path to the CLI script for pretraining
        jobs_dir: Path,         # Path to the directory for the job files
        logs_dir: Path,         # Path to the directory for the log files
        outs_dir: Path,         # Path to the directory for the output files
        storage_dir: Path,      # Path to the storage directory
        modules: list[str]      # List of modules to load
        ) -> None: 

    ss = 1 if stride_series else 0
 
    if job_name is None:
        job_name = f"{dataset}_wl{window_length}_ts{window_time_stride}_ps{window_patt_stride}_ss{ss}_{arch}"
    
    job_file = jobs_dir / (job_name + ".job")
    log_file = logs_dir / (job_name + ".log")
    out_file = outs_dir / (job_name + ".out")
    err_file = outs_dir / (job_name + ".err")

    cli_command = f"python {str(cli_script)} "
    cli_args = ["arch", "dataset",  
        "batch_size", "val_size",
        "window_length", "stride_series", 
        "window_time_stride", "window_patt_stride",
        "log_file", "storage_dir", "random_state"]

    for var in locals():
        if var in [cli_args]:
            cli_command += f"--{var} {str(locals()[var])} "

    with job_file.open(mode="w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        f.write(f"#SBATCH --account={account}\n")
        f.write(f"#SBATCH --partition={partition}\n")
        # if gpu:
        #     f.write(f"#SBATCH --gres=gpu:rtx8000:1\n")
        f.write(f"#SBATCH --nodes=1\n")
        f.write(f"#SBATCH --ntasks-per-node=1\n")
        f.write(f"#SBATCH --cpus-per-task={str(cpu)}\n")
        f.write(f"#SBATCH --mem={str(mem)}G\n")
        if email is not None:
            f.write(f"#SBATCH --mail-type=END\n")
            f.write(f"#SBATCH --mail-user={email}\n")
        f.write(f"#SBATCH -o {out_file}\n")
        f.write(f"#SBATCH -e {err_file}\n")
        for module in modules:
            f.write(f"module load {module}\n")
        f.write(f"source {str(venv_path)}\n")
        f.write(cli_command)

    subprocess.run(["sbatch", str(job_file)], capture_output=True)