
#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Common functions for the experiments. """

# pl imports
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

# in-package imports
from s3ts.models.wrapper import WrapperModel
from s3ts.legacy.modules import DFDataModule

# default pl settings
default_pl_kwargs: dict = {
    "default_root_dir": "training",
    "accelerator": "auto",
    "seed": 42
}

# metrics settings
metric_settings: dict = {
    "reg": {"all": ["mse", "r2"], "target": "mse", "mode": "min"},
    "cls": {"all": ["acc", "f1", "auroc"], "target": "acc", "mode": "max"}
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def create_model(
        
        ) -> WrapperModel:
    

    pass

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def test_model(
        dm: DFDataModule,        
        model: WrapperModel,
        pl_kwargs: dict = default_pl_kwargs
        ) -> dict:
    
    # choose metrics
    metrics = metric_settings[model.target]

    # set up the trainer   
    tr = Trainer(default_root_dir=pl_kwargs["default_root_dir"],  
        accelerator=pl_kwargs["accelerator"])
    
    # test the model
    data = tr.test(model, datamodule=dm)

    return {f"test_{m}": data[0][f"test_{m}"] for m in metrics}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def train_model(
        dm: DFDataModule, 
        model: WrapperModel,
        max_epochs: int,
        pl_kwargs: dict = default_pl_kwargs,
        ) -> tuple[WrapperModel, dict]:
    
    # reset the random seed
    seed_everything(pl_kwargs["seed"], workers=True)

    # choose metrics
    metrics = metric_settings[model.target]

    # set up the trainer
    ckpt = ModelCheckpoint(monitor=metrics["target"], mode=metrics["mode"])    
    tr = Trainer(default_root_dir=pl_kwargs["default_root_dir"],  
    accelerator=pl_kwargs["accelerator"], callbacks=[ckpt], max_epochs=max_epochs,
    logger=TensorBoardLogger(save_dir=pl_kwargs["default_root_dir"], name=model.name))

    # train the model
    tr.fit(model=model, datamodule=dm)

    # load the best weights
    model = model.load_from_checkpoint(ckpt.best_model_path)

    # run the validation with the final weights
    data = tr.validate(model, datamodule=dm)

    return model, {f"test_{m}": data[0][f"test_{m}"] for m in metrics}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #