from pathlib import Path

ATLAS = {
    "time"        : None,
    "account"     : "bcam-exclusive",
    "partition"   : "bcam-exclusive",
    "cpu"         : 6,
    "mem"         : 12,
    # "gres"        :  "gpu:rtx8000:1",
    "gres"        : None,
    # "email"       : "rcoterillo@bcamath.org",
    "email"       : None,
    "venv"        : Path("/scratch/rcoterillo/s3ts/s3ts_env/bin/activate"),
    "script"      : Path("/scratch/rcoterillo/s3ts/pret_cli.py"),
    "storage_dir" : Path("/scratch/rcoterillo/s3ts/storage"),
    "jobs_dir"    : Path("jobs/").absolute(),
    "logs_dir"    : Path("logs/").absolute(),
    "outs_dir"    : Path("outputs/").absolute(),
    "modules"     : [
        "CUDA/11.3.1",
        "cuDNN/cuDNN/8.2.1.32-CUDA-11.3.1",       
        "Python/3.10.4-GCCcore-11.3.0"
    ]
}

HIPATIA_MEDIUM = {
    "time"        : "6:00:00",
    "account"     : None,
    "partition"   : "medium",
    "cpu"         : 6,
    "mem"         : 12,
    # "gres"        :  "gpu:rtx8000:1",
    "gres"        : None,
    # "email"       : "rcoterillo@bcamath.org",
    "email"       : None,
    "venv"        : Path("/workspace/scratch/users/rcoterillo/s3ts/s3ts_env/bin/activate"),
    "script"      : Path("/workspace/scratch/users/rcoterillo/s3ts/pret_cli.py"),
    "storage_dir" : Path("/workspace/scratch/users/rcoterillo/s3ts/storage"),
    "jobs_dir"    : Path("jobs/").absolute(),
    "logs_dir"    : Path("logs/").absolute(),
    "outs_dir"    : Path("outputs/").absolute(),
    "modules"     : [
        "Python/3.9.5-GCCcore-10.3.0"
    ]
}

HIPATIA_LARGE = {
    "time"        : "5-00:00:00",
    "account"     : None,
    "partition"   : "large",
    "cpu"         : 6,
    "mem"         : 12,
    # "gres"        :  "gpu:rtx8000:1",
    "gres"        : None,
    # "email"       : "rcoterillo@bcamath.org",
    "email"       : None,
    "venv"        : Path("/workspace/scratch/users/rcoterillo/s3ts/s3ts_env/bin/activate"),
    "script"      : Path("/workspace/scratch/users/rcoterillo/s3ts/pret_cli.py"),
    "storage_dir" : Path("/workspace/scratch/users/rcoterillo/s3ts/storage"),
    "jobs_dir"    : Path("jobs/").absolute(),
    "logs_dir"    : Path("logs/").absolute(),
    "outs_dir"    : Path("outputs/").absolute(),
    "modules"     : [
        "Python/3.9.5-GCCcore-10.3.0"
    ]
}