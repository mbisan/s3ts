# package imports
from pathlib import Path
import subprocess

def sbatch_hook(
        # ~~~~~~~~~~~~~ Compulsory CLI Parameters ~~~~~~~~~~~~~
        dataset: str,
        mode: str,
        arch: str,
        # ~~~~~~~~~~~~~ HPC Profile Settings ~~~~~~~~~~~~~
        email: str,             # Email to send notifications
        cpu: int,               # Number of CPUs 
        mem: int,               # GB of RAM memory
        gres: str,              # Resources to use
        partition: str,         # Partition to use
        venv_path: Path,        # Path to the virtual environment
        cli_script: Path,       # Path to the CLI script
        jobs_dir: Path,         # Path to the directory for the job files
        logs_dir: Path,         # Path to the directory for the log files
        outs_dir: Path,         # Path to the directory for the output files
        modules: list[str],     # List of modules to load
        # ~~~~~~~~~~~~~ HPC Optional Settings ~~~~~~~~~~~~~
        job_name: str = None,   # Name of the job
        account: str = None,    # Account to charge
        time: str = None,       # Time limit for the job
        # ~~~~~~~~~~~~~ Optional CLI Parameters ~~~~~~~~~~~~~
        use_pretrain: bool = None,
        pretrain_mode: bool = None,
        rho_dfs: float = None,
        window_length: int = None,
        stride_series: bool = None,
        window_time_stride: int = None,
        window_patt_stride: int = None,
        num_encoder_feats: int = None,
        num_decoder_feats: int = None,
        exc: int = None, 
        train_event_mult: int = None,
        train_strat_size: int = None,
        test_sts_length: int = None,
        pret_sts_length: int = None,
        batch_size: int = None,
        val_size: float = None,
        max_epochs: int = None,
        learning_rate: float = None,
        random_state: int = None,
        cv_rep: int = None,
        log_file: Path = None,
        res_fname: str = None,
        train_dir: Path = None,
        storage_dir: Path = None,
        num_workers: int = None,
        ) -> None: 
    
    # Job name and output files
    ss = 1 if stride_series else 0
    if job_name is None:
        job_name = f"{dataset}_wl{window_length}_ts{window_time_stride}_ps{window_patt_stride}_ss{ss}_{arch}"
    
    # Ensure folders exist
    for folder in [jobs_dir, logs_dir, outs_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    job_file = jobs_dir / (job_name + ".job")
    log_file = logs_dir / (job_name + ".log")
    out_file = outs_dir / (job_name + ".out")
    err_file = outs_dir / (job_name + ".err")
    
    # CLI command
    params = locals()
    cli_params1 = ["dataset", "mode", "arch", 
        "rho_dfs", "window_length", "stride_series", 
        "window_time_stride", "window_patt_stride", 
        "num_encoder_feats", "num_decoder_feats", 
        "exc", "train_event_mult", "train_strat_size", 
        "test_sts_length", "pret_sts_length", 
        "batch_size", "val_size", "max_epochs", 
        "learning_rate", "random_state", "cv_rep", 
        "log_file", "res_fname", "train_dir", 
        "storage_dir", "num_workers"]
    cli_params2 = ["use_pretrain", "pretrain_mode"]
    cli_command = f"python {str(cli_script)} "
    for var in params:
        if var in cli_params1 and params[var] is not None:
            cli_command += f"--{var} {str(params[var])} "
        if var in cli_params2 and params[var] is not None and params[var]:
            cli_command += f"--{var} "
   
    # Write job file
    with job_file.open(mode="w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        if account is not None:
            f.write(f"#SBATCH --account={account}\n")
        if time is not None:
            f.write(f"#SBATCH --time={time}\n")
        f.write(f"#SBATCH --partition={partition}\n")
        if gres is not None:
            f.write(f"#SBATCH --gres={gres}\n")
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