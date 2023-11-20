


def main_loop(
        dataset: str,
        mode: str,
        arch: str,
        use_pretrain: bool,
        pretrain_mode: bool,
        # Model parameters
        rho_dfs: float,
        window_length: int,
        stride_series: bool,
        window_time_stride: int,
        window_patt_stride: int,
        num_encoder_feats: int,
        num_decoder_feats: int,
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
        num_workers: int,
        train_exc_limit: int = None,
        ):
    
    # ~~~~~~~~~~~ Create folders ~~~~~~~~~~~~

    start_time = time.perf_counter()

    for fold in ["datasets", "results", "encoders"]:
        path = storage_dir / fold
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    # Logging setup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if log_file is not None:
        log.basicConfig(filename=log_file, level=log.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        log.basicConfig(stream=sys.stdout, level=log.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S')

    # ~~~~~~~~~~~~ Sanity checks ~~~~~~~~~~~~

    log.info("Performing sanity checks...")

    # Check encoder parameters
    if mode not in ["ts", "df", "gf"]:
        raise ValueError(f"Invalid representation: {mode}")
    if arch not in ["nn", "rnn", "cnn", "res", "tcn", "dfn"]:
        raise ValueError(f"Invalid architecture: {arch}")
    valid_combinations = {"ts": ["nn", "rnn", "cnn", "res", "tcn"], 
            "df": ["cnn", "res", "tcn", "dfn"], "gf": ["cnn", "res", "tcn", "dfn"]}
    if arch not in valid_combinations[mode]:
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
