#!/usr/bin/env python

"""
Script that automates pretraining in a SBATCH queue.
"""

from s3ts.presets import HIPATIA_MEDIUM, HIPATIA_LARGE, ATLAS
from s3ts.hooks import sbatch_hook

# Common settings
# ~~~~~~~~~~~~~~~~~~~~~~~
PRETRAIN_ENCODERS = False         # Pretrain the DF encoders
TIME_DIL = True                   # Time dilation
PATT_STR = False                  # Pattern stride
SELF_SUP = False                  # Self-supervised pretraining
# ~~~~~~~~~~~~~~~~~~~~~~~
DATASETS = [ # Datasets
    "CBF", #"GunPoint", "Plane", "SyntheticControl"                                           
]
ARCHS = { # Architectures
    "DF": ["CNN", "ResNet"],
    "TS": ["RNN", "CNN", "ResNet"],
}
# ~~~~~~~~~~~~~~~~~~~~~~~
WINDOW_LENGTH_DF: int = 10                          # Window length for DF
WINDOW_LENGTHS_TS: list[int] = [10, 30, 50, 70]     # Window length for TS                   

WINDOW_TIME_STRIDES: list[int] = [1, 3, 5, 7]       # Window time stride
WINDOW_PATT_STRIDES: list[int] = [2, 3, 5]          # Window pattern stride
# ~~~~~~~~~~~~~~~~~~~~~~~
RHO_DFS: float = 0.1                # Memory parameter for DF
BATCH_SIZE: bool = 128              # Batch size
VAL_SIZE: float = 0.25              # Validation size
NUM_ENCODER_FEATS: int = 32         # Number of encoder features
NUM_DECODER_FEATS: int = 64         # Number of decoder features
# ~~~~~~~~~~~~~~~~~~~~~~~
EVENTS_PER_CLASS = 32               # Number of events per class
EVENT_LIMITERS = [8, 16, 32]        # Event limiters
TRAIN_EVENT_MULT = 4                # Training events multiplier
TRAIN_STRAT_SIZE = 2                # Training stratification size
TEST_STS_LENGTH = 200               # Number of events for testing
PRET_STS_LENGTH = 1000              # Number of events for pretraining
# ~~~~~~~~~~~~~~~~~~~~~~~
MAX_EPOCHS_PRE: int = 60            # Pre-training epochs
MAX_EPOCHS_TRA: int = 120           # Training epochs
LEARNING_RATE: float = 1E-4         # Learning rate
# ~~~~~~~~~~~~~~~~~~~~~~~
RES_FNAME = "results.csv"           # Results filename
# ~~~~~~~~~~~~~~~~~~~~~~~
RANDOM_STATE = 0                    # Random state
CV_REPS = 5                         # Number of cross-validation repetitions
# ~~~~~~~~~~~~~~~~~~~~~~~
SHARED_ARGS = {"rho_dfs": RHO_DFS, "exc": EVENTS_PER_CLASS, 
    "batch_size": BATCH_SIZE, "val_size": VAL_SIZE,
    "num_encoder_feats": NUM_ENCODER_FEATS, 
    "num_decoder_feats": NUM_DECODER_FEATS,
    "learning_rate": LEARNING_RATE,
    "train_event_mult": TRAIN_EVENT_MULT, 
    "train_strat_size": TRAIN_STRAT_SIZE,
    "test_sts_length": TEST_STS_LENGTH, 
    "pret_sts_length": PRET_STS_LENGTH,
    "random_state": RANDOM_STATE}
SHARED_ARGS = {**SHARED_ARGS, **HIPATIA_MEDIUM}

# Pretrain Loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if PRETRAIN_ENCODERS:
    for arch in ARCHS["DF"]:
        for dataset in DATASETS:
            res_fname = f"results_pretrain_{arch}_{dataset}.csv"
            wlen = WINDOW_LENGTH_DF
            for wts in WINDOW_TIME_STRIDES:
                # Full series
                sbatch_hook(dataset=dataset, mode="DF", arch=arch,
                    use_pretrain=False, pretrain_mode=True,
                    window_length=wlen, stride_series=False,
                    window_time_stride=wts, window_patt_stride=1,
                    max_epochs=MAX_EPOCHS_PRE, cv_rep=0, 
                    res_fname=res_fname, **SHARED_ARGS)
                if wts != 1:
                    # Strided series
                    sbatch_hook(dataset=dataset, mode="DF", arch=arch,
                        use_pretrain=False, pretrain_mode=True,
                        window_length=wlen, stride_series=True,
                        window_time_stride=wts, window_patt_stride=1,
                        max_epochs=MAX_EPOCHS_PRE, cv_rep=0, 
                        res_fname=res_fname, **SHARED_ARGS)
            for wps in WINDOW_PATT_STRIDES:
                # Full series
                sbatch_hook(dataset=dataset, mode="DF", arch=arch,
                    use_pretrain=False, pretrain_mode=True,
                    window_length=wlen, stride_series=False,
                    window_time_stride=7, window_patt_stride=wps,
                    max_epochs=MAX_EPOCHS_PRE, cv_rep=0, 
                    res_fname=res_fname, **SHARED_ARGS)

# Training Loop for Section 5.1: Time Dilation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if TIME_DIL:
    for cv_rep in range(CV_REPS):

        # Do the TS training
        mode = "TS"
        for arch in ARCHS[mode]:
            for dataset in DATASETS:
                res_fname = f"results_{mode}_{arch}_{dataset}_cv{cv_rep}.csv"
                for wlen in WINDOW_LENGTHS_TS:
                    sbatch_hook(dataset=dataset, mode=mode, arch=arch,
                        use_pretrain=False, pretrain_mode=False,
                        window_length=wlen, stride_series=False,
                        window_time_stride=1, window_patt_stride=1,
                        max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                        res_fname=res_fname, **SHARED_ARGS)
                    
        # Do the DF training
        mode = "DF"
        for arch in ARCHS[mode]:
            for dataset in DATASETS:

                res_fname = f"results_{mode}_{arch}_{dataset}_cv{cv_rep}.csv"

                wlen = WINDOW_LENGTH_DF
                wps  = 1

                for wts in WINDOW_TIME_STRIDES:
                    sbatch_hook(dataset=dataset, mode=mode, arch=arch,
                        use_pretrain=False, pretrain_mode=False,
                        window_length=wlen, stride_series=False,
                        window_time_stride=wts, window_patt_stride=wps,
                        max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                        res_fname=res_fname, **SHARED_ARGS)

# Training Loop for Section 5.2: Pattern Stride
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if PATT_STR:
    for cv_rep in range(CV_REPS):
                    
        # Do the DF training
        mode = "DF"
        for arch in ARCHS[mode]:
            for dataset in DATASETS:

                res_fname = f"results_{mode}_{arch}_{dataset}_cv{cv_rep}.csv"

                wlen = WINDOW_LENGTH_DF
                wts  = 7

                for wps in WINDOW_PATT_STRIDES:
                    # Full series
                    sbatch_hook(dataset=dataset, mode=mode, arch=arch,
                        use_pretrain=False, pretrain_mode=False,
                        window_length=wlen, stride_series=False,
                        window_time_stride=wts, window_patt_stride=wps, 
                        max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                        res_fname=res_fname, **SHARED_ARGS)


# Training Loop for Section 5.3: Self-Supervised Pretraining
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#TODO
if SELF_SUP:
    for cv_rep in range(CV_REPS):
                    
        # Do the DF training
        mode = "DF"
        for arch in ARCHS[mode]:
            for dataset in DATASETS:

                res_fname = f"results_{mode}_{arch}_{dataset}_cv{cv_rep}.csv"
                
                for ev_lim in EVENT_LIMITERS:

                    wlen = WINDOW_LENGTH_DF
                    wts  = 7
                    wps  = 2 

                    # Full series
                    sbatch_hook(dataset=dataset, mode=mode, arch=arch,
                        use_pretrain=True, pretrain_mode=False,
                        window_length=wlen, stride_series=False,
                        window_time_stride=wts, window_patt_stride=wps, 
                        max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                        res_fname=res_fname, **SHARED_ARGS)

                    # Full series
                    sbatch_hook(dataset=dataset, mode=mode, arch=arch,
                        use_pretrain=True, pretrain_mode=False,
                        window_length=wlen, stride_series=False,
                        window_time_stride=wts, window_patt_stride=wps, 
                        max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                        res_fname=res_fname, **SHARED_ARGS)
                    
                    # Strided series
                    sbatch_hook(dataset=dataset, mode=mode, arch=arch,
                        use_pretrain=True, pretrain_mode=False,
                        window_length=wlen, stride_series=True,
                        window_time_stride=wts, window_patt_stride=wps, 
                        max_epochs=MAX_EPOCHS_TRA, cv_rep=cv_rep, 
                        res_fname=res_fname, **SHARED_ARGS)


