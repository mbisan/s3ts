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

# datamodule
from s3ts.data.modules import DFDataModule

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
        repr: str, arch: str,
        target: str, dm: DFDataModule,
        learning_rate: float,
        decoder_feats: int = 64,
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
        
        self.n_classes = dm.n_classes
        self.n_patterns = dm.n_patterns
        self.l_patterns = dm.n_patterns
        self.window_length = dm.window_length
        self.window_time_stride = dm.window_time_stride
        self.window_pattern_stride = dm.window_pattern_stride
        self.stride_series = dm.stride_series

        self.learning_rate = learning_rate
        self.decoder_feats = decoder_feats

        # Save hyperparameters
        self.save_hyperparameters(learning_rate=learning_rate, repr=repr, arch=arch, target=target, 
            window_length=self.window_length, window_time_stride=self.window_time_stride, 
            window_pattern_stride=self.window_pattern_stride, stride_series=self.stride_series)
        
        # Create the encoder
        self.encoder = nn.Sequential()
        if repr == "DF":
            ref_size, channels = self.l_patterns, self.n_patterns
        elif repr == "TS":
            ref_size, channels = 1, 1 
        self.encoder.add_module("encoder", encoder_arch(
            ref_size=ref_size, channels=channels, 
            window_size=self.window_length))

        # Determine the number of decoder input features
        in_feats = self.window_length if self.repr == "DF" else self.encoder.get_output_shape()
      
        # Add the metrics depending on the target
        self.decoder = nn.Sequential()
        if self.target == "cls":
            
            # Determine the number of output features
            out_feats = self.n_classes

            # Add the decoder modules
            if self.repr == "DF":
                self.decoder.add_module("lstm", LSTMDecoder(
                        in_features = in_feats,
                        hid_features = decoder_feats,
                        out_features = out_feats))
            elif self.repr == "TS":
                self.decoder.add_module("linear", LinearDecoder(
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
                    in_features = in_feats,
                    hid_features = decoder_feats**2,
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
        output = self(frames)

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
        if stage == "train":
            self.log(f"{stage}_loss", loss, sync_dist=True)
            if self.target == "cls":
                self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)
                self.log(f"{stage}_f1", f1, prog_bar=True, sync_dist=True)
            elif self.target == "reg":
                self.log(f"{stage}_mse", mse, prog_bar=True, sync_dist=True)
                self.log(f"{stage}_r2", r2, prog_bar=True, sync_dist=True)

        # Return the loss
        return loss.to(torch.float32)

    def training_step(self, batch, batch_idx):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch, batch_idx):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        """ Test step. """
        return self._inner_step(batch, stage="test")

    # EPOCH END
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _custom_epoch_end(self, step_outputs: list[torch.Tensor], stage: str):

        """ Custom epoch end for the training, validation and testing. """

        # Metrics to compute
        if self.target == "cls":
            metrics = ["acc", "f1"]
            if stage != "train":
                metrics.append("auroc")
        elif self.target == "reg":
            metrics = ["mse", "r2"]
        
        # Compute, log and print the metrics
        if stage == "val":
            print("")
        print(f"\n\n  ~~ {stage} stats ~~")
        for metric in metrics:
            mstring = f"{stage}_{metric}"
            val = self.__getattr__(mstring).compute()
            if stage == "train":
                self.log("epoch_" + mstring, val, sync_dist=True)
            else:
                self.log(mstring, val, sync_dist=True)
            self.__getattr__(mstring).reset()
            print(f"{mstring}: {val:.4f}")
        print("")

    def training_epoch_end(self, training_step_outputs):
        """ Actions to carry out at the end of each training epoch. """
        self._custom_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        """ Actions to carry out at the end of each validation epoch. """
        self._custom_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        """ Actions to carry out at the end of each test epoch. """
        self._custom_epoch_end(test_step_outputs, "test")


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