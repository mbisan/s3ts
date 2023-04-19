"""
Base Convolutional Classification Model

@author Raúl Coterillo
@version 2023-01
"""

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
from s3ts.models.encoders.frames.ResNet import ResNet_DFS
from s3ts.models.encoders.frames.CNN import CNN_DFS
from s3ts.models.encoders.series.ResNet import ResNet_TS
from s3ts.models.encoders.series.CNN import CNN_TS
from s3ts.models.encoders.series.RNN import RNN_TS

# numpy
import logging as log
import numpy as np

# ========================================================= #
#                     MULTITASK MODEL                       #
# ========================================================= #

class WrapperModel(LightningModule):

    def __init__(self,
        repr: str, 
        arch: str,
        target: str,
        n_classes: int,
        n_patterns: int,
        l_patterns: int,
        window_length: int,
        window_time_stride: int,
        window_patt_stride: int,
        stride_series: bool,
        encoder_feats: int,
        decoder_feats: int,
        learning_rate: float,
        ):

        """ Wrapper for the PyTorch models used in the experiments. """

        super().__init__()
        
        self.encoder_dict = {"TS": {"RNN": RNN_TS, "CNN": CNN_TS, "ResNet": ResNet_TS}, 
                        "DF": {"CNN": CNN_DFS, "ResNet": ResNet_DFS}}
        
        # Check encoder parameters
        if repr not in ["DF", "TS"]:
            raise ValueError(f"Invalid representation: {repr}")
        if arch not in ["RNN", "CNN", "ResNet"]:
            raise ValueError(f"Invalid architecture: {arch}")
        if arch not in self.encoder_dict[repr]:
            raise ValueError(f"Architecture {arch} not available for representation {repr}.")
        encoder_arch = self.encoder_dict[repr][arch]

        # Check decoder parameters
        if target not in ["cls", "reg"]:
            raise ValueError(f"Invalid target: {target}")
        
        # Gather model parameters
        self.repr = repr
        self.arch = arch
        self.target = target
        self.n_classes = n_classes
        self.n_patterns = n_patterns
        self.l_patterns = l_patterns
        self.window_length = window_length
        self.window_time_stride = window_time_stride
        self.window_patt_stride = window_patt_stride
        self.stride_series = stride_series
        self.encoder_feats = encoder_feats
        self.decoder_feats = decoder_feats
        self.learning_rate = learning_rate

        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create the encoder
        self.encoder = nn.Sequential()
        if repr == "DF":
            ref_size = len(np.arange(self.l_patterns)[::self.window_patt_stride])
            channels = self.n_patterns
        elif repr == "TS":
            ref_size, channels = 1, 1 
        self.encoder.add_module("encoder", encoder_arch(
            ref_size=ref_size, channels=channels, 
            window_size=self.window_length))

        # Determine the number of decoder input features
        inp_feats = self.encoder[0].lat_channels if self.repr == "DF" else 1
        inp_size = self.window_length if self.repr == "DF" else self.encoder.encoder.get_output_shape()
        
        # Add the metrics depending on the target
        self.decoder = nn.Sequential()
        if self.target == "cls":
            
            # Determine the number of output features
            out_feats = self.n_classes

            # Add the decoder modules
            if self.repr == "DF":
                self.decoder.add_module("lstm", LSTMDecoder(
                        inp_size=inp_size,
                        inp_features=inp_feats,
                        hid_features = decoder_feats,
                        out_features = out_feats))
            elif self.repr == "TS":
                self.decoder.add_module("linear", LinearDecoder(
                        in_features = inp_size,
                        hid_features = decoder_feats,
                        out_features = out_feats))
            self.decoder.add_module("softmax", nn.Softmax())
            
            # Add the metrics
            for phase in ["train", "val", "test"]: 
                self.__setattr__(f"{phase}_acc", tm.Accuracy(num_classes=out_feats, task="multiclass"))
                self.__setattr__(f"{phase}_f1",  tm.F1Score(num_classes=out_feats, task="multiclass"))
                if phase != "train":
                    self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=out_feats, task="multiclass"))

        elif self.target == "reg":

            # Determine the number of output features
            out_feats = self.window_length if self.stride_series else self.window_length*self.window_time_stride
            
            # Add the decoder modules
            self.decoder.add_module("lstm", LSTMDecoder(
                        inp_size=inp_size,
                        inp_features=inp_feats,
                        hid_features = decoder_feats,
                        out_features = out_feats))

            # Add the metrics
            for phase in ["train", "val", "test"]:
                self.__setattr__(f"{phase}_mse", tm.MeanSquaredError(squared=False))
                self.__setattr__(f"{phase}_r2",  tm.R2Score(num_outputs=out_feats))

    def forward(self, frame):
        """ Forward pass. """
        out = self.encoder(frame)
        out = self.decoder(out)
        return out

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _inner_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Unpack the batch from the dataloader
        frames, series, label = batch

        # Forward pass
        if self.repr == "DF":
            output = self(frames)
        elif self.repr == "TS":
            output = self(series)

        # Compute the loss and metrics
        if self.target == "cls":
            loss = F.cross_entropy(output, label.to(torch.float32))
            acc = self.__getattr__(f"{stage}_acc")(output, torch.argmax(label, dim=1))
            f1  = self.__getattr__(f"{stage}_f1")(output, torch.argmax(label, dim=1))
            if stage != "train":
                auroc = self.__getattr__(f"{stage}_auroc")(output, torch.argmax(label, dim=1))  
        elif self.target == "reg":
            loss = F.mse_loss(output, series)
            mse = self.__getattr__(f"{stage}_mse")(output, series)
            r2 = self.__getattr__(f"{stage}_r2")(output, series)

        # Log the loss and metrics
        on_step = True if stage == "train" else False

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=True, sync_dist=True, logger=True)
        if self.target == "cls":
            self.log(f"{stage}_acc", acc, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, logger=True)
            self.log(f"{stage}_f1", f1, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, logger=True)
            self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, logger=True)
        elif self.target == "reg":
            self.log(f"{stage}_mse", mse, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, logger=True)
            self.log(f"{stage}_r2", r2, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, logger=True)

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

        if self.target == "cls":
            mode, monitor = "max", "val_acc"
        elif self.target == "reg":
            mode, monitor = "min", "val_mse"

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode=mode, factor=np.sqrt(0.1), patience=5, min_lr=0.5e-7),
                "interval": "epoch",
                "monitor": monitor,
                "frequency": 10
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }