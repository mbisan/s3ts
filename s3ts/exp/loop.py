import numpy as np

def train_test_splits(X: np.ndarray, Y: np.ndarray, 
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
        test_idx = np.setdiff1d(idx, train_idx)

        yield train_idx, test_idx

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def setup_train_dm(
        X: np.ndarray, Y: np.ndarray, 
        mode: str, patterns: np.ndarray,
        train_idx: np.ndarray, test_idx: np.ndarray, 
        test_sts_length: int, 
        train_event_limit: int,
        train_strat_size: int,
        train_event_mult: int, 
        rho_dfs: float, 
        batch_size: int, val_size: float,
        window_length: int,
        stride_series: bool,
        window_time_stride: int, 
        window_patt_stride: int,
        random_state: int,
        num_workers: int = mp.cpu_count()//2,
        ) -> DFDataModule:

    """ Sets up the training DataModule."""

    # Get the train, test and medoid events
    X_train, Y_train = X[train_idx,:], Y[train_idx]
    X_test, Y_test = X[test_idx,:], Y[test_idx]

    # Validate the inputs
    event_length = X_train.shape[1]     # Get the length of the time series

    # Check there is the same numbe of classes in train and test
    if len(np.unique(Y_train)) != len(np.unique(Y_test)):
        raise ValueError("The number of classes in train and test must be the same.")

    # Check there is the same number of events in each class in train
    if len(np.unique(np.unique(Y_train, return_counts=True)[1])) != 1:
        raise ValueError("The number of events in each class in train must be the same.")

    # Check the number of events in each class in train is a multiple of the stratification size
    if len(Y_train)%train_strat_size != 0:
        raise ValueError("The number of events in each class in train must be a multiple of the stratification size.")

    STS_nev_train = len(train_idx)*train_event_mult
    STS_nev_test = test_sts_length

    log.info("Generating the train STS")
    STS_train, SCS_train = compute_STS(X_train, Y_train,        
        shift_limits=True, STS_events=STS_nev_train, 
        mode="stratified", event_strat_size=train_strat_size,
        random_state=random_state, add_first_event=True)
    
    log.info("Generating the test STS")
    STS_test, SCS_test = compute_STS(X_test, Y_test,                
        shift_limits=True, STS_events=STS_nev_test, mode="random", 
        random_state=random_state, add_first_event=True)

    if mode == "df" or mode == "ts":
        log.info("Computing the train DM")
        DM_train = compute_DM_optim(STS_train, patterns, rho_dfs)
        log.info("Computing the test DM")
        DM_test = compute_DM_optim(STS_test, patterns, rho_dfs)
    elif mode == "gf":
        log.info("Computing the train GM")
        DM_train = compute_GM_optim(STS_train, patterns)
        log.info("Computing the test GM")
        DM_test = compute_GM_optim(STS_test, patterns)

    # Remove the first sample from the STSs
    STS_train, STS_test = STS_train[event_length:], STS_test[event_length:]
    SCS_train, SCS_test = SCS_train[event_length:], SCS_test[event_length:]
    DM_train, DM_test = DM_train[:,:,event_length:], DM_test[:,:,event_length:]

    # Remove events according to train_event_limit
    limit_idx = event_length*train_event_limit
    STS_train, STS_test = STS_train[:limit_idx], STS_test[:limit_idx]
    SCS_train, SCS_test = SCS_train[:limit_idx], SCS_test[:limit_idx]
    DM_train, DM_test = DM_train[:,:,:limit_idx], DM_test[:,:,:limit_idx]

    # Return the DataModule
    return DFDataModule(
        X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
        STS_train=STS_train, SCS_train=SCS_train, DM_train=DM_train,
        STS_test=STS_test, SCS_test=SCS_test, DM_test=DM_test,
        event_length=event_length, patterns=patterns,
        batch_size=batch_size, val_size=val_size, 
        stride_series=stride_series, window_length=window_length,
        window_time_stride=window_time_stride, 
        window_patt_stride=window_patt_stride,
        random_state=random_state, 
        num_workers=num_workers)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def setup_pretrain_dm(
        X: np.ndarray, Y: np.ndarray,
        mode: str, patterns: np.ndarray, 
        sts_length: int, rho_dfs: float, 
        batch_size: int, val_size: float,
        window_length: int,
        window_time_stride: int, 
        window_patt_stride: int,
        random_state: int,
        stride_series: bool = False,
        num_workers: int = mp.cpu_count()//2,
        ) -> DFDataModule:

    """ Prepare the pretraining DataModule. """

    # Get the length of the time series
    event_length = X.shape[1]
    
    log.info("Generating the pretrain STS")
    STS_pret, SCS_pret = compute_STS(X, Y,                
        shift_limits=True, STS_events=sts_length, 
        mode="random", random_state=random_state, 
        add_first_event=True)

    if mode == "df":
        log.info("Computing the pretrain DM")
        DM_pret = compute_DM_optim(STS_pret, patterns, rho_dfs)
    elif mode == "gf":
        log.info("Computing the pretrain GM")
        DM_pret = compute_GM_optim(STS_pret, patterns)

    # Remove the first sample from the STSs
    STS_pret = STS_pret[event_length:]
    SCS_pret = SCS_pret[event_length:]
    DM_pret = DM_pret[:,:,event_length:]

    # Return the DataModule
    return DFDataModule(
        X_train=X, Y_train=Y, X_test=None, Y_test=None,
        STS_train=STS_pret, SCS_train=SCS_pret, DM_train=DM_pret,
        STS_test=None, SCS_test=None, DM_test=None,
        event_length=event_length, patterns=patterns,
        batch_size=batch_size, val_size=val_size,
        stride_series=stride_series, 
        window_length=window_length,
        window_time_stride=window_time_stride, 
        window_patt_stride=window_patt_stride,
        random_state=random_state, 
        num_workers=num_workers)




def experiment_loop(
        # control parameters
        dset: str,
        dsrc: str,
        arch: str,
        pret: bool,
        pret_mode: bool,
        # data parameters
        rho_dfs: float,
        wdw_len: int,
        wdw_str: int,
        str_sts: bool,
        # model complexity
        enc_feats: int,
        dec_feats: int,
        # Training parameters
        exc: int,
        train_event_mult: int,
        train_strat_size: int,
        test_sts_length: int,
        pret_sts_length: int,
        batch_size: int,
        val_size: float,
        max_epochs: int,
        learning_rate: float,
        random_state: int,
        cv_rep: int,
        # Paths
        log_file: Path,
        res_fname: str,
        train_dir: Path,
        storage_dir: Path,
        train_exc_limit: int = None,
        ):
    
    # ~~~~~~~~~~~ Create folders ~~~~~~~~~~~~

    start_time = time.perf_counter()

    for fold in ["datasets", "results", "encoders"]:
        path = storage_dir / fold
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


    # ~~~~~~~~~~~~ Sanity checks ~~~~~~~~~~~~

    log.info("Performing sanity checks...")

    # Check encoder parameters
    if dsrc not in ["ts", "df", "gf"]:
        raise ValueError(f"Invalid representation: {mode}")
    if arch not in ["nn", "rnn", "cnn", "res"]:
        raise ValueError(f"Invalid architecture: {arch}")
    valid_combinations = {"ts": ["nn", "rnn", "cnn", "res"], 
            "df": ["cnn", "res"], "gf": ["cnn", "res"]}
    if arch not in valid_combinations[dsrc]:
        raise ValueError(f"Architecture {arch} not available for representation {mode}.")

    # Check stride_series is false if mode is TS
    if mode == "ts" and stride_series:
        raise ValueError("Stride series must be False for ts mode.")

    # Check all window parameters are positive integers
    for val in [window_length, window_time_stride, window_patt_stride]:
        if val < 1 or not isinstance(val, int):
            raise ValueError("Window paramters must be positive integers.")
    
    # Check mode is 'df' if use_pretrain is True
    if (use_pretrain and mode not in ["df", "gf"]) or (pretrain_mode and mode not in ["df", "gf"]):
        raise ValueError("Pretraining is only available for df/gf modes.")

    # Check pretrain_mode and use_pretrain are not both True
    if use_pretrain and pretrain_mode:
        raise ValueError("'pretrain_mode' is a previous step to 'use_pretrain', so they cannot be both True.")

    # Get the path to the encoder
    assert isinstance(stride_series, bool)
    ss = 1 if stride_series else 0
    enc_name = f"{dataset}_{mode}_{arch}_wl{window_length}_ts{window_time_stride}_ps{window_patt_stride}_ss{ss}"
    encoder_path = storage_dir / "encoders" / (enc_name + ".pt")

    if use_pretrain or pretrain_mode:
        log.info(f"encoder_path: {encoder_path}")

    # If not in pretrain_mode and use_pretrain, check the encoder exists
    if use_pretrain and not pretrain_mode:
        if not encoder_path.exists():
            raise ValueError("Encoder not found. Please run pretrain mode first.")
    
    # If pretrain_mode, check the encoder does not exist already
    if pretrain_mode:
        if encoder_path.exists():
            raise ValueError("Encoder already exists. Please delete it before running pretrain mode.")
    
    # If train_exc_limit is None, set it to exc
    if train_exc_limit is None:
        train_exc_limit = exc

    # Check the train_exc_limit is positive and not greater than exc
    if train_exc_limit > exc:
        raise ValueError("The train event limit cannot be greater than the number of events per class.")
    if train_exc_limit < 1:
        raise ValueError("The train event limit must be positive.")

    # If use_pretrain is False, set encoder_path to None
    if not pretrain_mode and not use_pretrain:
        encoder_path = None
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Download the dataset or load it from storage
    # TODO add support for external datasets
    X, Y, medoids, medoid_idx = download_dataset(dataset=dataset, storage_dir=storage_dir)

    # If arch is 'nn', set the window length to the length of the samples
    if arch == "nn":
        window_length = X.shape[1]

    if pretrain_mode:
        # Get directory and version
        directory = train_dir / "pretrain" / f"{dataset}_{mode}_{arch}"
        version = f"wlen{window_length}_stride{ss}" +\
            f"_wtst{window_time_stride}_wpst{window_patt_stride}" +\
            f"_val{val_size}_me{max_epochs}_bs{batch_size}" +\
            f"_stsl{pret_sts_length}_lr{learning_rate}_rs{random_state}"
        # Setup the data module
        dm = setup_pretrain_dm(X, Y, patterns=medoids, 
            sts_length=pret_sts_length,rho_dfs=rho_dfs, 
            batch_size=batch_size, val_size=val_size,
            window_length=window_length, 
            stride_series=stride_series, 
            window_time_stride=window_time_stride, 
            window_patt_stride=window_patt_stride, 
            random_state=random_state,
            num_workers=num_workers,
            mode=mode)
    else:
        # Get the train and test idx for the current CV repetition
        for j, (train_idx, test_idx) in enumerate(train_test_splits(X, Y, exc=exc, nreps=cv_rep+1, random_state=random_state)):
            if j == cv_rep:
                break
        # Get directory and version
        directory = train_dir / "finetune" / f"{dataset}_{mode}_{arch}"
        if mode == "df":
            version = f"exc{exc}_avev{train_exc_limit}_tstrat{train_strat_size}" +\
                f"_tmult{train_event_mult}_ststest{test_sts_length}" +\
                f"_wlen{window_length}_stride{ss}" +\
                f"_wtst{window_time_stride}_wpst{window_patt_stride}" +\
                f"_val{val_size}_me{max_epochs}_bs{batch_size}" +\
                f"_lr{learning_rate}_rs{random_state}_cv{cv_rep}"
        else:
            version = f"exc{exc}_avev{train_exc_limit}_tstrat{train_strat_size}" +\
                f"_tmult{train_event_mult}_ststest{test_sts_length}" +\
                f"_wlen{window_length}" +\
                f"_val{val_size}_me{max_epochs}_bs{batch_size}" +\
                f"_lr{learning_rate}_rs{random_state}_cv{cv_rep}"
            
        # Calculate event limits
        train_event_total = int(exc*len(np.unique(Y[train_idx]))*train_event_mult)
        train_event_limit = int(train_exc_limit*len(np.unique(Y[train_idx]))*train_event_mult)

        # Setup the data module
        dm = setup_train_dm(X=X, Y=Y, patterns=medoids,
            train_idx=train_idx, test_idx=test_idx,
            test_sts_length=test_sts_length,
            train_event_mult=train_event_mult,
            train_strat_size=train_strat_size,
            train_event_limit=train_event_limit,
            batch_size=batch_size, val_size=val_size,               
            rho_dfs=rho_dfs, window_length=window_length, 
            window_time_stride=window_time_stride,
            window_patt_stride=window_patt_stride,
            stride_series=stride_series,
            random_state=random_state,
            num_workers=num_workers,
            mode=mode)
    
    dm_time = time.perf_counter()
    
    # Train the model
    data, model = run_model(
        pretrain_mode=pretrain_mode, version=version,
        dataset=dataset, mode=mode, arch=arch, dm=dm, 
        directory=directory, max_epochs=max_epochs,
        learning_rate=learning_rate, encoder_path=encoder_path,
        num_encoder_feats=num_encoder_feats,
        num_decoder_feats=num_decoder_feats,
        random_state=random_state, cv_rep=cv_rep)
    
    train_time = time.perf_counter()
    
    if not pretrain_mode:
        data["exc"] = exc 
        data["train_exc_limit"] = train_exc_limit
        data["train_strat_size"] = train_strat_size
        data["train_event_mult"] = train_event_mult
        data["nevents_test"] = dm.STS_test_events
        data["nevents_train_lim"] = train_event_limit
        data["nevents_train_tot"] = train_event_total

    # Log times
    data["time_dm"] = dm_time - start_time
    data["time_train"] = train_time - dm_time
    data["time_total"] = train_time - start_time

    # Save the results
    save_results(data, res_fname=res_fname, storage_dir=storage_dir)
