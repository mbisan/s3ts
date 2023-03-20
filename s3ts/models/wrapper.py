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

# numpy
import logging as log
import numpy as np

# ========================================================= #
#                     MULTITASK MODEL                       #
# ========================================================= #

class WrapperModel(LightningModule):

    def __init__(self,      
        n_labels: int,
        n_patterns: int, 
        l_patterns: int,
        window_length: int,
        lab_shifts: list[int],
        arch: type[LightningModule],
        approach: str = "lstm",
        target: str = "cls",
        learning_rate: float = 1e-4,
        ):

        super().__init__()
        self.save_hyperparameters()

        self.n_labels = n_labels
        self.n_shifts = len(lab_shifts)
        self.n_patterns = n_patterns
        self.l_patterns = l_patterns
        self.window_length = window_length
        self.learning_rate = learning_rate
        self.approach = approach
        self.target = target
        
        # encoder
        
        if approach not in ["lstm", "linear", "old"]:
            raise NotImplementedError()
        if target not in ["cls", "reg"]:
            raise NotImplementedError()
        
        self.decoder_features = 256 

        # decoder
        # self.decoder = LinearDecoder(in_features=embedding_size, hid_features=embedding_size//2, 
        #     out_features=n_labels*self.n_shifts)

        self.encoder: LightningModule = arch(               # encoder
                ref_size=l_patterns, 
                channels=n_patterns, 
                window_size=window_length)
        
        if self.approach == "lstm":

            if self.target == "cls":
                self.decoder = nn.Sequential(LSTMDecoder(   # decoder
                    in_features = window_length,
                    hid_features = self.decoder_features // 4,
                    out_features = n_labels
                    ), nn.Softmax())    
                for phase in ["train", "val", "test"]:      # metrics
                    self.__setattr__(f"{phase}_acc", tm.Accuracy(num_classes=n_labels, task="multiclass"))
                    self.__setattr__(f"{phase}_f1",  tm.F1Score(num_classes=n_labels, task="multiclass"))
                    if phase != "train":
                        self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=n_labels, task="multiclass"))
            elif self.target == "reg":
                self.decoder = LSTMDecoder(                 # decoder
                    in_features = window_length,
                    hid_features = self.decoder_features // 4,
                    out_features = window_length)
                for phase in ["train", "val", "test"]:      # metrics
                    self.__setattr__(f"{phase}_mse", tm.MeanSquaredError(squared=False))
                    self.__setattr__(f"{phase}_r2",  tm.R2Score(num_outputs=window_length))
    
        elif self.approach == "linear":
            
            if self.target == "cls":
                self.decoder = nn.Sequential(                   # decoder
                    nn.Flatten(), LinearDecoder(     
                    hid_features = self.decoder_features,
                    hid_layers = 2, out_features = n_labels
                    ), nn.Softmax())    
                for phase in ["train", "val", "test"]:          # metrics
                    self.__setattr__(f"{phase}_acc", tm.Accuracy(num_classes=n_labels, task="multiclass"))
                    self.__setattr__(f"{phase}_f1",  tm.F1Score(num_classes=n_labels, task="multiclass"))
                    if phase != "train":
                        self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=n_labels, task="multiclass"))
            elif self.target == "reg":
                self.decoder = nn.Sequential(                   # decoder
                    nn.Flatten(), LinearDecoder(                   
                    hid_features = self.decoder_features,
                    hid_layers = 2, out_features = window_length))
                for phase in ["train", "val", "test"]:          # metrics
                    self.__setattr__(f"{phase}_mse", tm.MeanSquaredError(squared=False))
                    self.__setattr__(f"{phase}_r2",  tm.R2Score(num_outputs=window_length))

        elif self.approach == "old":
            
            self.encoder = nn.Sequential(arch(                  # encoder
                ref_size=l_patterns, channels=n_patterns, 
                window_size=window_length),
                nn.Flatten(), nn.LazyLinear(out_features=256),
                nn.Linear(in_features=128, out_features=256))
            
            if self.target == "cls":
                self.decoder = nn.Sequential(LinearDecoder(     # decoder
                    hid_features = self.decoder_features,
                    out_features = n_labels
                    ), nn.Softmax())    
                for phase in ["train", "val", "test"]:          # metrics
                    self.__setattr__(f"{phase}_acc", tm.Accuracy(num_classes=n_labels, task="multiclass"))
                    self.__setattr__(f"{phase}_f1",  tm.F1Score(num_classes=n_labels, task="multiclass"))
                    if phase != "train":
                        self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=n_labels, task="multiclass"))
            elif self.target == "reg":
                self.decoder = LinearDecoder(                   # decoder
                    hid_features = self.lstm_features,
                    out_features = window_length)
                for phase in ["train", "val", "test"]:          # metrics
                    self.__setattr__(f"{phase}_mse", tm.MeanSquaredError(squared=False))
                    self.__setattr__(f"{phase}_r2",  tm.R2Score(num_outputs=window_length))



    # FORWARD
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def forward(self, frame):

        """ Use for inference only (separate from training_step)"""

        out = self.encoder(frame)
        out = self.decoder(out)
        return out

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _inner_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: str = None):

        """ Common actions for training, test and val steps. """

        # x[0] is the time series
        # x[1] are the sim frames
        
        frames, series, label = batch

        output = self(frames)

        # accumulate and return metrics for logging
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

        if stage == "train":
            self.log(f"{stage}_loss", loss, sync_dist=True)
            if self.target == "cls":
                self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)
                self.log(f"{stage}_f1", f1, prog_bar=True, sync_dist=True)
            elif self.target == "reg":
                self.log(f"{stage}_mse", mse, prog_bar=True, sync_dist=True)
                self.log(f"{stage}_r2", r2, prog_bar=True, sync_dist=True)

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

        """ Common actions for validation and test epoch ends. """

        # metrics to analyze

        if self.target == "cls":
            metrics = ["acc", "f1"]
            if stage != "train":
                metrics.append("auroc")
        elif self.target == "reg":
            metrics = ["mse", "r2"]
        
        # task flags
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

    # OPTIMIZERS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def configure_optimizers(self):

        """ Define optimizers and LR schedulers. """

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