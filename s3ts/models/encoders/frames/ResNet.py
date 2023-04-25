#lightning
from pytorch_lightning import LightningModule

from torch import nn
import torch

class ResidualBlock(LightningModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=8, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same"),
            nn.BatchNorm2d(out_channels)
        )

        self.block.apply(ResidualBlock.initialize_weights)
        self.shortcut.apply(ResidualBlock.initialize_weights)

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
        elif hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor):
        block = self.block(x)
        shortcut = self.shortcut(x)

        block = torch.add(block, shortcut)
        return nn.functional.relu(block)


class ResNet_DFS(LightningModule):

    def __init__(self, ref_size: int , channels: int, window_size: int):
        super().__init__()

        self.channels = channels
        self.n_feature_maps = 16

        self.model = nn.Sequential(
            ResidualBlock(in_channels=channels, out_channels=self.n_feature_maps),
            ResidualBlock(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2),
            ResidualBlock(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2),
            ResidualBlock(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 4),
            #nn.AvgPool2d((ref_size, window_size))
            #nn.AvgPool2d((5, 1))
        )
        self.calculate_output_shape(ref_size, channels, window_size)

    def calculate_output_shape(self, ref_size:int, channels:int, window_size:int):
        x = torch.rand((1, channels, ref_size, window_size))
        shp: torch.Size = self(x).shape
        self.lat_channels = shp[1]
        self.lat_patt_length = shp[2]
        self.lat_time_length = shp[3]

    def get_output_shape(self):
        return (len(self.model._modules) -1) * self.n_feature_maps

    def forward(self, x):
        out = self.model(x.float())
        return out

    