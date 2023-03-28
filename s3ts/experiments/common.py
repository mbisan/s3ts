#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Common functions for the experiments. """

# models / modules
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning import Trainer

# data processing stuff
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from scipy.spatial import distance_matrix

# in-package imports
from s3ts.models.wrapper import WrapperModel
from s3ts.data.modules import DFDataModule
from s3ts.data.oesm import compute_DM_optim

# standard library
from pathlib import Path
import logging as log

# basics
import pandas as pd
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def train_pretest_split(X: np.ndarray, Y: np.ndarray, 
        exc: int, nreps: int, random_state: int):

    """ Splits the dataset into train and pretest sets.
    Selects sxc events per class for the train set and the rest for the pretest set.
    
    Parameters
    ----------
    X : np.ndarray
        The time series dataset.
    Y : np.ndarray
        The labels of the time series dataset.
    sxc : int
        The number of events per class in the train set.
    nreps : int
        The number of different splits.
    random_state : int
        Random state for the RNG.
    """

    # Check the shape of the dataset and labels match
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of events in the dataset and labels must be the same.")

    # Check the number of events per class is not larger than the total number of events
    if exc > X.shape[0]:
        raise ValueError("The number of events per class cannot be larger than the total number of events.")
    
    # Check the number of events per class is not larger than the number of events per class
    if exc > np.unique(Y, return_counts=True)[1].min():
        raise ValueError("The number of events per class cannot be larger than the minimum number of events per class.")

    idx = np.arange(X.shape[0])
    rng = np.random.default_rng(random_state)

    for _ in range(nreps):
        
        train_idx = []
        for c in np.unique(Y):
            train_idx.append(rng.choice(idx, size=exc, p=(Y==c).astype(int)/sum(Y==c), replace=False))
        train_idx = np.concatenate(train_idx)
        pretest_idx = np.setdiff1d(idx, train_idx)

        yield train_idx, pretest_idx

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_medoids(
        X: np.ndarray, 
        Y: np.ndarray,
        distance_type: str = 'dtw'
    ) -> tuple[np.ndarray, np.ndarray]: 

    """ Computes the medoids of the classes in the dataset. 
    
    Parameters
    ----------
    X : np.ndarray
        The time series dataset.
    Y : np.ndarray
        The labels of the time series dataset.
    distance_type : str, optional
        The distance type to use, by default 'dtw'
    """

    # Check the distance type
    if distance_type not in ["euclidean", "dtw"]:
        raise ValueError("The distance type must be either 'euclidean' or 'dtw'.")
    
    # Check the shape of the dataset and labels match
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of events in the dataset and labels must be the same.")

    # Get the number of classes
    n_classes = len(np.unique(Y))
    
    # Get the length of the time series
    s_length = X.shape[1]

    # Initialize the arrays
    medoids = np.empty((n_classes, s_length), dtype=float)
    medoid_ids = np.empty(n_classes, dtype=int)
    
    # Find the medoids for each class
    for i, y in enumerate(np.unique(Y)):

        # Get the events of the class
        index = np.argwhere(Y == y)
        Xy = X[index, :]

        # ...using Euclidean distance        
        if distance_type == "euclidean":
            medoid_idx = np.argmin(distance_matrix(Xy.squeeze(), Xy.squeeze()).sum(axis=0))
            medoids[i,:] = Xy[medoid_idx,:]
            medoid_ids[i] = index[medoid_idx]

        # ...using Dynamic Time Warping (DTW)
        if distance_type == "dtw":
            if Xy.shape[0] > 1:
                tskm = TimeSeriesKMedoids(n_clusters=1, init_algorithm="forgy", metric="dtw")
                tskm.fit(Xy)
                medoids[i,:] = tskm.cluster_centers_.squeeze()
                medoid_ids[i] = np.where(np.all(Xy.squeeze() == medoids[i,:], axis=1))[0][0]
            else:
                medoids[i,:] = Xy.squeeze()
                medoid_ids[i] = np.where(np.all(Xy.squeeze() == medoids[i,:], axis=1))[0][0]

    # Return the medoids and their indices
    return medoids, medoid_ids

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_STS(
        X: np.ndarray, 
        Y: np.ndarray,
        STS_events: int,
        shift_limits: bool,
        mode: str = "random",
        random_state: int = 0,
        event_strat_size: int = 4,
        add_first_sample: bool = False,
        ) -> tuple[np.ndarray, np.ndarray]:

    """ Generates a Streaming Time Series (STS) from a given dataset. """

    # Check the shape of the dataset and labels match
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of events in the dataset and labels must be the same.")

    # Set the random state for reproducibility
    rng = np.random.default_rng(seed=random_state)
    
    # Get the number of classes
    n_classes = len(np.unique(Y))
    
    # Get the length of the time series
    s_length = X.shape[1]
    
    # Get the number of events
    n_events = X.shape[0]

    # Get the length of the final STS
    STS_length = STS_events*s_length

    # Do some logging
    log.info(f"Number of events: {n_events}")
    log.info(f"Length of events: {s_length}")
    log.info(f"Number of classes: {n_classes}")
    log.info(f"Class ratios: {np.unique(Y, return_counts=True)[1]/n_events}")
    log.info(f"Length of STS: {STS_length}")

    # Initialize the arrays
    if add_first_sample:
        STS = np.empty(STS_length+s_length, dtype=np.float32)
        SCS = np.empty(STS_length+s_length, dtype=np.int8)
        random_idx = rng.integers(0, n_events)
        STS[0:s_length] = X[random_idx,:]
        SCS[0:s_length] = Y[random_idx]
    else:
        STS = np.empty(STS_length, dtype=np.float32)
        SCS = np.empty(STS_length, dtype=np.int8)

    # Generate the STS 
    if mode == "random":
        for s in range(STS_events):

            random_idx = rng.integers(0, n_events)
            s = s+1 if add_first_sample else s

            # Calculate shift so that sample ends match
            shift = STS[s-1] - X[random_idx,0] if shift_limits else 0

            STS[s*s_length:(s+1)*s_length] = X[random_idx,:] + shift
            SCS[s*s_length:(s+1)*s_length] = Y[random_idx]

    if mode == "stratified":
        
        exc =  n_events//n_classes

        if exc%event_strat_size != 0:
            raise ValueError("The number of events per class must be a multiple of the event stratification size.")
    
        if STS_events%exc != 0:
            raise ValueError("The number of events in the STS must be a multiple of the number of events per class.")

        event_idx = np.arange(X.shape[0])
        
        clist = []
        for c in np.unique(Y):
            Yc_idx = event_idx[Y==c]
            rng.shuffle(Yc_idx)
            clist.append(np.reshape(Yc_idx, (-1, event_strat_size)))

        strats = np.concatenate(clist, axis=1)
        n_repeats = STS_events // n_events

        cidx = 1 if add_first_sample else 0
        for strat in range(strats.shape[0]):
            for _ in range(n_repeats):
                for s in rng.permutation(strats[strat,:]):

                    # Calculate shift so that sample ends match
                    shift = STS[cidx-1] - X[s,0] if shift_limits else 0

                    STS[cidx*s_length:(cidx+1)*s_length] = X[s,:] + shift
                    SCS[cidx*s_length:(cidx+1)*s_length] = Y[s]

                    # Calculate shift so that sample ends match
                    cidx += 1

    # Normalize the STS
    STS = (STS - np.mean(STS))/np.std(STS)

    # Return the STS and the SCS
    return STS, SCS

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def prepare_dm(
        dataset: str, 
        X_train: np.ndarray, X_pretest: np.ndarray, 
        Y_train: np.ndarray, Y_pretest: np.ndarray,
        train_events_per_class: int,
        train_event_multiplier: int,
        pret_event_multiplier: int,
        test_event_multiplier: int,
        rho_dfs: float, batch_size: int, val_size: float,
        window_length: int, stride_series: bool,
        window_time_stride: int, window_patt_stride: int,
        fold_number: int, random_state: int,
        num_workers: int = 4,
        use_cache: bool = True,
        pattern_type: str = "medoids",
        cache_dir: Path = Path("cache"),
        ) -> DFDataModule:

    """ Prepare the data module for training/pretraining/testing. """

    # Validate the inputs
    n_classes = len(np.unique(Y_train))     # Get the number of classes
    s_length = X_train.shape[1]             # Get the length of the time series

    # Check the pattern type is valid
    valid_patterns = ["medoids"]
    if pattern_type not in valid_patterns:
        raise ValueError(f"patterns must be one of {valid_patterns}")

    # Check there is the same numbe of classes in train and test
    if len(np.unique(Y_train)) != len(np.unique(Y_pretest)):
        raise ValueError("The number of classes in train and test must be the same.")
    
    # Check the number of events per class in train
    if np.unique(Y_train, return_counts=True)[1].min() < train_events_per_class:
        raise ValueError(f"The number of events per class in the train set must be at least {train_events_per_class}.")

    # Check the number of events per class in pretest
    if np.unique(Y_pretest, return_counts=True)[1].min() < train_events_per_class*2:
        raise ValueError(f"The number of events per class in the pretest set must be at least {train_events_per_class*2}.")

    # Generate filenames for the cache files using the parametersç
    multiplier_str = f"{train_events_per_class}sxc_{train_event_multiplier}tramult_{pret_event_multiplier}pretmult_{test_event_multiplier}testmult"
    cache_file = cache_dir / f"{dataset}_{multiplier_str}_{pattern_type}_fold{fold_number}_rs{random_state}.npz"

    STS_train_events = int(train_events_per_class*n_classes)*train_event_multiplier
    STS_pret_events = int(train_events_per_class*n_classes)*pret_event_multiplier
    STS_test_events = int(train_events_per_class*n_classes)*test_event_multiplier

    # If the cache file exists, load everything from there
    if use_cache and cache_file.exists():

        log.info(f"Loading data from cache file {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        STS_tra, SCS_tra = data["STS_tra"], data["SCS_tra"]
        STS_pre, SCS_pre = data["STS_pre"], data["SCS_pre"]
        DM_tra, DM_pre = data["DM_tra"], data["DM_pre"]
        patterns = data["patterns"]

    else:

        # Generate the STSs
        STS_tra, SCS_tra = compute_STS(X_train, Y_train,        # Generate train STS  
            shift_limits=True, STS_events=STS_train_events, mode="stratified", 
            random_state=random_state, add_first_sample=True)
        STS_pre, SCS_pre = compute_STS(X_pretest, Y_pretest,    # Generate pretest STS 
            shift_limits=True, STS_events=STS_pret_events+STS_test_events, mode="random", 
            random_state=random_state, add_first_sample=True)

        # Generate the patterns for the DMs
        if pattern_type == "medoids":
            log.info("Selecting the medoids from the train data")
            medoids, medoid_ids = compute_medoids(X_train, Y_train, distance_type="dtw")
            patterns = medoids
  
         # Generate the DMs
        log.info("Computing the training DM")
        DM_tra = compute_DM_optim(STS_tra, patterns, rho_dfs)
        log.info("Computing the pretrain/test DM")
        DM_pre = compute_DM_optim(STS_pre, patterns, rho_dfs)

        # Remove the first sample from the STSs
        STS_tra = STS_tra[s_length:]
        STS_pre = STS_pre[s_length:]
        SCS_tra = SCS_tra[s_length:]
        SCS_pre = SCS_pre[s_length:]
        DM_tra = DM_tra[:,:,s_length:]
        DM_pre = DM_pre[:,:,s_length:]

        # Save the data to the cache file
        np.savez(cache_file, patterns=patterns,
            STS_tra=STS_tra, SCS_tra=SCS_tra, DM_tra=DM_tra,
            STS_pre=STS_pre, SCS_pre=SCS_pre, DM_pre=DM_pre)

    # Return the DataModule
    return DFDataModule(
        STS_tra=STS_tra, SCS_tra=SCS_tra, DM_tra=DM_tra,
        STS_pre=STS_pre, SCS_pre=SCS_pre, DM_pre=DM_pre,
        STS_train_events=STS_train_events, 
        STS_pret_events=STS_pret_events,
        STS_test_events=STS_test_events,
        sample_length=s_length, patterns=patterns,
        batch_size=batch_size, val_size=val_size, 
        pretrain = False, window_length=window_length,
        stride_series=stride_series,
        window_time_stride=window_time_stride, 
        window_patt_stride=window_patt_stride,
        random_state=random_state, num_workers=num_workers)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_model(
        dataset: str, repr: str, arch: str,
        dm: DFDataModule, pretrain: bool, fold_number: int, 
        directory: Path, label: str,
        max_epoch_pre: int, max_epoch_tgt: int,
        train_events_per_class: int, train_event_multiplier: int,
        pret_event_multiplier: int, test_event_multiplier: int,
        learning_rate: float, random_state: int = 0, 
        ) -> tuple[pd.DataFrame, WrapperModel, ModelCheckpoint]:

    def _setup_trainer(max_epochs: int, stop_metric: str, 
            stop_mode: str, pretrain: bool) -> tuple[Trainer, ModelCheckpoint]:
        version = f"{label}_pretrain" if pretrain else label
        # Create the callbacks
        checkpoint = ModelCheckpoint(monitor=stop_metric, mode=stop_mode)    
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        #early_stop = EarlyStopping(monitor=stop_metric, mode=stop_mode, patience=20)
        callbacks = [lr_monitor, checkpoint]#, early_stop]
        # Creathe the loggers
        tb_logger = TensorBoardLogger(save_dir=directory, name="logs", version=version)
        csv_logger = CSVLogger(save_dir=directory, name="logs", version=version)
        loggers = [tb_logger, csv_logger]
        # Create the trainer
        return Trainer(default_root_dir=directory,  accelerator="auto", devices="auto",
        logger=loggers, callbacks=callbacks,
        max_epochs=max_epochs,  benchmark=True, deterministic=False, 
        log_every_n_steps=1, check_val_every_n_epoch=1), checkpoint

    cls_metrics = ["acc", "f1", "auroc"]
    reg_metrics = ["mse", "r2"]

    # Pretrain the model if needed
    if pretrain:
        
        log.info("Pretraining the encoder...")

        # Create the model, trainer and checkpoint
        pre_model = WrapperModel(repr=repr, arch=arch, target="reg",
            n_classes=dm.n_classes, window_length=dm.window_length, 
            n_patterns=dm.n_patterns, l_patterns=dm.l_patterns,
            window_time_stride=dm.window_time_stride, window_patt_stride=dm.window_patt_stride,
            stride_series=dm.stride_series, encoder_feats=32, decoder_feats=64,
            learning_rate=learning_rate)
        pre_trainer, pre_ckpt = _setup_trainer(max_epoch_pre, "val_mse", "min", True)

        # Configure the datamodule
        dm.pretrain = True

        # Perform the pretraining
        pre_trainer.fit(pre_model, datamodule=dm)

        # Load the best checkpoint
        pre_model = pre_model.load_from_checkpoint(pre_ckpt.best_model_paths)

    # Configure the datamodule
    dm.pretrain = False

    log.info("Training the target model...")

    # Create the model, trainer and checkpoint
    tgt_model = WrapperModel(repr=repr, arch=arch, target="cls",
        n_classes=dm.n_classes, window_length=dm.window_length, 
        n_patterns=dm.n_patterns, l_patterns=dm.l_patterns,
        window_time_stride=dm.window_time_stride, window_patt_stride=dm.window_patt_stride,
        stride_series=dm.stride_series, encoder_feats=32, decoder_feats=64,
        learning_rate=learning_rate)
    tgt_trainer, tgt_ckpt = _setup_trainer(max_epoch_tgt, "val_acc", "max", False)

    # Load encoder if needed
    if pretrain:
        tgt_model.encoder = pre_model.encoder

    # Train the model
    tgt_trainer.fit(tgt_model, datamodule=dm)

    # Load the best checkpoint
    tgt_model = tgt_model.load_from_checkpoint(tgt_ckpt.best_model_path)

    # Save the experiment settings and results
    res = pd.Series(dtype="object")
    res["dataset"] = dataset
    res["arch"], res["repr"] = arch, repr
    res["pretrain"], res["fold_number"], res["random_state"] = pretrain, fold_number, random_state
    res["batch_size"], res["stride_series"], res["window_length"] = dm.batch_size, dm.stride_series, dm.window_length
    res["window_time_stride"], res["window_pattern_stride"] = dm.window_time_stride, dm.window_pattern_stride
    res["train_events_per_class"] = train_events_per_class
    res["train_event_multiplier"] = train_event_multiplier
    res["nevents_train"] = dm.av_train_events
    res["pret_event_multiplier"] = pret_event_multiplier
    res["nevents_pret"] = dm.av_pret_events if pretrain else 0
    res["test_event_multiplier"] = test_event_multiplier
    res["nevents_test"] = dm.av_test_events
    res["tgt_best_model"] = tgt_ckpt.best_model_path
    res["tgt_train_csv"] = str(directory  / "logs" / label / "metrics.csv")
    res["tgt_nepochs"] = int(tgt_ckpt.best_model_path.split("/")[5][6:].split("-")[0])
    train_res_val  = tgt_trainer.validate(tgt_model, datamodule=dm)
    train_res_test = tgt_trainer.test(tgt_model, datamodule=dm)
    for m in cls_metrics:
        res[f"target_val_{m}"]  = train_res_val[0][f"val_{m}"]
        res[f"target_test_{m}"] = train_res_test[0][f"test_{m}"]
    if pretrain:
        dm.pretrain = True
        pret_res = pre_trainer.validate(pre_model, datamodule=dm)
        dm.pretrain = False
        for m in reg_metrics:
            res[f"pre_val_{m}"] = pret_res[0][f"val_{m}"]
        res["pre_best_model"] = tgt_ckpt.best_model_path
        res["pre_train_csv"] = str(directory  / "logs" / label / "metrics.csv")
        res["pre_nepochs"] = int(pre_ckpt.best_model_path.split("/")[5][6:].split("-")[0])

    # Convert to DataFrame
    res = res.to_frame().transpose().copy()

    return res, tgt_model

def update_results_file(res_list: list[pd.DataFrame], new_res: pd.DataFrame, res_file: Path):

    # update results file
    res_list.append(new_res)
    log.info(f"Updating results file ({str(res_file)})")
    res_df = pd.concat(res_list, ignore_index=True)
    res_df.to_csv(res_file, index=False)

    return res_list