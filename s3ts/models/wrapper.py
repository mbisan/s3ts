#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Wrapper model for the deep learning models. """

from __future__ import annotations

# lightning
from s3ts.models.decoders.linear import LinearDecoder
from s3ts.models.decoders.lstm import LSTMDecoder
from pytorch_lightning import LightningModule

# base torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchmetrics as tm
import torch.nn as nn
import torch

# architectures
from s3ts.models.encoders.frames.CNN import CNN_IMG
from s3ts.models.encoders.frames.RES import RES_IMG

from s3ts.models.encoders.series.RNN import RNN_TS
from s3ts.models.encoders.series.CNN import CNN_TS
from s3ts.models.encoders.series.RES import RES_TS

# numpy
import logging as log
import numpy as np


encoder_dict = {"img": {"cnn": CNN_IMG, "res": RES_IMG},
    "ts": {"rnn": RNN_TS, "cnn": CNN_TS, "res": RES_TS}}

class WrapperModel(LightningModule):

    name: str           # model name
    dtype: str          # input dtype: ["img", "ts"]
    arch: str           # architecture: ["rnn", "cnn", "res"]
    task: str           # task type: ["cls", "reg"]

    wdw_len: int        # window length
    wdw_str: int        # window stride
    sts_str: bool       # stride the series too?

    n_dims: int         # number of STS dimensions
    n_classes: int      # number of classes
    n_patterns: int     # number of patterns
    l_patterns: int     # pattern size
    
    enc_feats: int      # encoder feature hyperparam
    dec_feats: int      # decoder feature hyperparam
    lr: float           # learning rate

    def __init__(self, name, dtype, arch, task,
        n_classes, n_patterns,l_patterns,
        wdw_len, wdw_str, sts_str,
        enc_feats, dec_feats, lr
        ) -> None:

        """ Wrapper for the PyTorch models used in the experiments. """

        # save parameters as attributes
        super(WrapperModel).__init__(), self.__dict__.update(locals())

        # select model architecture class
        enc_arch: LightningModule = encoder_dict[dtype][arch]

        # create encoder
        if dtype == "img":
            ref_size, channels = l_patterns, n_patterns
        elif dtype == "ts":
            ref_size, channels = 1, self.n_dims 
        self.encoder = enc_arch(channels=channels, ref_size=ref_size, 
            wdw_size=self.wdw_len, n_feature_maps=self.enc_feats)
        
        # create decoder
        shape: torch.Tensor = self.encoder.get_output_shape()
        inp_feats = torch.prod(torch.tensor(shape[1:]))
        if self.task == "cls":
            out_feats = self.n_classes
        elif self.task == "reg":
            out_feats = self.wdw_len if self.sts_str else self.wdw_len*self.wdw_str
        self.decoder = LinearDecoder(inp_feats=inp_feats, 
            hid_feats=dec_feats, out_feats=out_feats, hid_layers=2)

        # create softmax and flatten layerss
        self.flatten = nn.Flatten(start_dim=1)
        self.softmax = nn.Softmax()

        # create metrics
        if self.task == "cls":
            for phase in ["train", "val", "test"]: 
                self.__setattr__(f"{phase}_acc", tm.Accuracy(num_classes=out_feats, task="multiclass"))
                self.__setattr__(f"{phase}_f1",  tm.F1Score(num_classes=out_feats, task="multiclass"))
                if phase != "train":
                    self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=out_feats, task="multiclass"))
        elif self.task == "reg":
            for phase in ["train", "val", "test"]:
                self.__setattr__(f"{phase}_mse", tm.MeanSquaredError(squared=False))
                self.__setattr__(f"{phase}_r2",  tm.R2Score(num_outputs=out_feats))

    def forward(self, x: torch.Tensor):
        """ Forward pass. """
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.decoder(x)
        if self.task == "cls":
            x = self.softmax(x)
        return x

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Unpack the batch from the dataloader
        # frames, series, label = batch

        # Forward pass
        if self.dtype == "img":
            output = self(batch["frame"])
        elif self.dtype == "ts":
            output = self(torch.unsqueeze(batch["series"] , dim=1))

        # Compute the loss and metrics
        if self.task == "cls":
            loss = F.cross_entropy(output, batch["label"].to(torch.float32))
            acc = self.__getattr__(f"{stage}_acc")(output, torch.argmax(batch["label"], dim=1))
            f1  = self.__getattr__(f"{stage}_f1")(output, torch.argmax(batch["label"], dim=1))
            if stage != "train":
                auroc = self.__getattr__(f"{stage}_auroc")(output, torch.argmax(batch["label"], dim=1))  
        elif self.task == "reg":
            loss = F.mse_loss(output, batch[1])
            mse = self.__getattr__(f"{stage}_mse")(output,  batch["series"])
            r2 = self.__getattr__(f"{stage}_r2")(output,  batch["series"])

        # Log the loss and metrics
        on_step = True if stage == "train" else False
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)
        if self.task == "cls":
            self.log(f"{stage}_acc", acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log(f"{stage}_f1", f1, on_epoch=True, on_step=False, prog_bar=False, logger=True)
            if stage != "train":
                self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        elif self.task == "reg":
            self.log(f"{stage}_mse", mse, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log(f"{stage}_r2", r2, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        # Return the loss
        return loss.to(torch.float32)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """ Test step. """
        return self._inner_step(batch, stage="test")

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def configure_optimizers(self):

        """ Configure the optimizers. """

        mode = "max" if self.task == "cls" else "min"
        monitor = "val_acc" if self.task == "cls" else "val_mse"
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, 
                    mode=mode, factor=np.sqrt(0.1), patience=5, min_lr=0.5e-7),
                "interval": "epoch",
                "monitor": monitor,
                "frequency": 10
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }