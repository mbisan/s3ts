from pathlib import Path

BCAM_HIPATIA = {
    "account"     : "bcam-exclusive",
    "partition"   : "bcam-exclusive",
    "cpu"         : 6,
    "mem"         : 10,
    # "gres"        :  "gpu:rtx8000:1",
    "email"       : None,
    "venv"        : Path("/scratch/rcoterillo/s3ts/s3ts_env/bin/activate"),
    "script"      : Path("/scratch/rcoterillo/s3ts/pret_cli.py"),
    "storage_dir" : Path("/scratch/rcoterillo/s3ts/storage"),
    "jobs_dir"    : Path("jobs/").absolute(),
    "logs_dir"    : Path("logs/").absolute(),
    "outs_dir"    : Path("outputs/").absolute(),
    "storage_dir" : Path("/scratch/rcoterillo/s3ts/storage"),
    "modules"     : [
        "CUDA/11.3.1",
        "cuDNN/cuDNN/8.2.1.32-CUDA-11.3.1",       
        "Python/3.10.4-GCCcore-11.3.0"
    ]
}
