# lightning
from pytorch_lightning import LightningModule

import torch.nn as nn
import torch

class CNN_DFS(LightningModule):

    def __init__(self, ref_size:int, channels:int, window_size:int):
        super().__init__()

        self.channels = channels
        self.n_feature_maps = 32

        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=channels, out_channels=self.n_feature_maps // 2, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=self.n_feature_maps // 2, out_channels=self.n_feature_maps,
                      kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.AvgPool2d(2),

            nn.Dropout(0.35),

            nn.Conv2d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2,
                      kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=self.n_feature_maps * 2),
            nn.ReLU(),

            nn.Dropout(0.4),
            nn.Conv2d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 4,
                      kernel_size=3, padding='same'),
        )
        self.linear_1 = self.dynamic_linear((1, channels, ref_size, window_size))
        self.linear_2 = nn.Linear(in_features=self.n_feature_maps * 4, out_features=self.n_feature_maps * 8)

    @staticmethod
    def __str__() -> str:
        return "CNN_DFS"

    @staticmethod
    def __frames__() -> bool:
        return True

    def dynamic_linear(self, image_dim):
        x = torch.rand(*(image_dim))
        features = self.model(x.float())
        flat = features.view(features.size(0), -1)
        return nn.Linear(in_features=flat.size(1), out_features=self.n_feature_maps * 4)

    def get_output_shape(self):
        return self.n_feature_maps * 8

    def forward(self, x):
        features = self.model(x.float())
        flat = features.view(features.size(0), -1)
        lin_1 = self.linear_1(flat)
        return self.linear_2(lin_1)