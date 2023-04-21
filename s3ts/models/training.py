#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Common functions for the experiments. """

# models / modules
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
import torch

# in-package imports
from s3ts.models.wrapper import WrapperModel
from s3ts.data.modules import DFDataModule

# standard library
from pathlib import Path
import logging as log

# basics
import pandas as pd


CLS_METRICS = ["acc", "f1", "auroc"]


def setup_trainer(
        label: str,
        directory: Path,
        pretrain: bool,
        max_epochs: int, 
        stop_metric: str, 
        stop_mode: str, 
        ) -> tuple[Trainer, ModelCheckpoint]:
    
    """ Setup the trainer. """
    
    # Create the callbacks
    checkpoint = ModelCheckpoint(monitor=stop_metric, mode=stop_mode)    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    #early_stop = EarlyStopping(monitor=stop_metric, mode=stop_mode, patience=20)
    callbacks = [lr_monitor, checkpoint]#, early_stop]
    # Create the loggers
    tb_logger = TensorBoardLogger(save_dir=directory, name="logs", version=label)
    csv_logger = CSVLogger(save_dir=directory, name="logs", version=label)
    loggers = [tb_logger, csv_logger]
    # Create the trainer
    return Trainer(default_root_dir=directory,  accelerator="auto", devices="auto",
    logger=loggers, callbacks=callbacks,
    max_epochs=max_epochs,  benchmark=True, deterministic=False, 
    log_every_n_steps=1, check_val_every_n_epoch=1), checkpoint

def pretrain_encoder(dataset: str, repr: str, arch: str, dm: DFDataModule, 
        directory: Path, max_epoch: int, learning_rate: float, 
        storage_dir: Path, random_state: int = 0, 
        ) -> tuple[WrapperModel, ModelCheckpoint]:
    
    # Set the random seed
    seed_everything(random_state, workers=True)

    # Encoder path
    ss = 1 if dm.stride_series else 0
    enc_name1 = f"{dataset}sl{dm.STS_train_events}_me{max_epoch}_rs{random_state}"
    enc_name2 = f"_ss{ss}_wl{dm.window_length}_ts{dm.window_time_stride}_ps{dm.window_patt_stride}"
    encoder_name = enc_name1 + enc_name2
    encoder_path = storage_dir / "encoders" / (encoder_name + ".pt")

    # Check if the encoder is already trained
    if encoder_path.exists():
         # raise an error
        raise FileExistsError(f"The encoder '{encoder_name}' already exists in {storage_dir}")

    metrics = ["mse", "r2"]
    trainer, ckpt = setup_trainer(label=encoder_name, directory=directory,
        pretrain=True, max_epochs=max_epoch, stop_metric="val_mse", stop_mode="min")

    log.info("Pretraining the encoder...")
    # Create the model, trainer and checkpoint
    model = WrapperModel(repr=repr, arch=arch, target="reg",
        n_classes=dm.n_classes, window_length=dm.window_length, 
        n_patterns=dm.n_patterns, l_patterns=dm.l_patterns,
        window_time_stride=dm.window_time_stride, window_patt_stride=dm.window_patt_stride,
        stride_series=dm.stride_series, encoder_feats=32, decoder_feats=64,
        learning_rate=learning_rate)
    
    # uncomment whenever they fix the bugs lol
    # model: torch.Module = torch.compile(model, mode="reduce-overhead")
        
    # Perform the pretraining
    trainer: Trainer
    trainer.fit(model, datamodule=dm)

    # Load the best checkpoint
    ckpt: ModelCheckpoint
    model = model.load_from_checkpoint(ckpt.best_model_path)
    model.save_hyperparameters

    # Save the pretrained encoder
    torch.save(model.encoder, encoder_path)

    return model, ckpt

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
    
    # Set the random seed
    seed_everything(random_state, workers=True)

    def _setup_trainer(max_epochs: int, stop_metric: str, 
            stop_mode: str, pretrain: bool) -> tuple[Trainer, ModelCheckpoint]:
        version = f"{label}_pretrain" if pretrain else label
        # Create the callbacks
        checkpoint = ModelCheckpoint(monitor=stop_metric, mode=stop_mode)    
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        #early_stop = EarlyStopping(monitor=stop_metric, mode=stop_mode, patience=20)
        callbacks = [lr_monitor, checkpoint]#, early_stop]
        # Create the loggers
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
        torch.compile(pre_model, mode="reduce-overhead")
        pre_trainer, pre_ckpt = _setup_trainer(max_epoch_pre, "val_mse", "min", True)

        # Configure the datamodule
        dm.pretrain = True

        # Perform the pretraining
        pre_trainer.fit(pre_model, datamodule=dm)

        # Load the best checkpoint
        pre_model = pre_model.load_from_checkpoint(pre_ckpt.best_model_path)

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
    res["window_time_stride"], res["window_patt_stride"] = dm.window_time_stride, dm.window_patt_stride
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