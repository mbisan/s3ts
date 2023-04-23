
from pathlib import Path
import subprocess

def launch_pret_sbatch(
        # pretrain settings
        dataset: str, arch: str, 
        sts_length: int, rho_dfs: float,
        batch_size: int, val_size: float,
        max_epoch: int, learning_rate: float,
        window_length: int, stride_series: bool,
        window_time_stride: int, window_patt_stride: int,  
        random_state: int,
        # sbatch / hpc settings
        job_name: str, email: str,
        cpu: int, mem: int,
        account: str, partition: str,
        venv_path: Path,
        cli_script: Path,
        jobs_dir: Path,
        logs_dir: Path,
        outs_dir: Path,
        storage_dir: Path,
        modules: list[str]
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