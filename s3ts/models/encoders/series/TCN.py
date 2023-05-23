from pytorch_lightning import LightningModule
from torch.nn.utils import weight_norm
from torch import nn
import numpy as np
import torch

# https://arxiv.org/abs/1803.01271

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)    

class TCN_TS(LightningModule):

    """ Recurrent neural network (LSTM) for times series. """

    # def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        
    def __init__(self, channels: int, wdw_size: int, ref_size: int,
                 n_feature_maps: int = 32, dropout: float = 0.2):
        
        super(TCN_TS, self).__init__()

        # register parameters
        self.channels = channels
        self.wdw_size = wdw_size
        self.ref_size = 1 # here for compatibility
        self.n_feature_maps = n_feature_maps


        kernel_size = 3
        dilation_base = 2

        n = int(np.ceil(np.log2((wdw_size-1)*(dilation_base-1)/(kernel_size-1) +1)))
        layer_feats = [n_feature_maps]*n
        
        layers = []
        num_levels = len(layer_feats)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = channels if i == 0 else layer_feats[i-1]
            out_channels = layer_feats[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        layers.append(nn.AvgPool1d(kernel_size=(3)))
        self.network = nn.Sequential(*layers)

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape
    
    def forward(self, x):
        return self.network(x)
