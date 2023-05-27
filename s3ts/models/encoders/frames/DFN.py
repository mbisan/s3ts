from pytorch_lightning import LightningModule
from torch.nn.utils import weight_norm
from torch import nn
import numpy as np
import torch

# https://arxiv.org/abs/1803.01271

class Chomp2d(nn.Module):
    def __init__(self, chomp_size: tuple):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
        return x[:,:, self.chomp_size[0]:, :-self.chomp_size[1]].contiguous()

class TemporalBlock(nn.Module):
    
    """ Temporal block for the DFN model. """

    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: tuple[int], 
                stride, dilation, padding, dropout=0.2):
        
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(in_channels=in_channels, 
                                           out_channels=out_channels, 
                                           kernel_size=kernel_size,
                                           stride=stride, 
                                           padding=padding, 
                                           dilation=dilation))
        self.chomp1 = Chomp2d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(in_channels=out_channels, 
                                           out_channels=out_channels, 
                                           kernel_size=kernel_size,
                                           stride=stride, 
                                           padding=padding, 
                                           dilation=dilation))
        self.chomp2 = Chomp2d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                            kernel_size=1) if in_channels != out_channels else None
        
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

class DFN_DF(LightningModule):

    """ FrameNet network for dissimilarity frames. """

    # def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        
    def __init__(self, channels: int, wdw_size: int, ref_size: int,
                 n_feature_maps: int = 32, dropout: float = 0.2,
                 orientation: str = "series"):
        
        super(DFN_DF, self).__init__()

        # register parameters
        self.channels = channels
        self.wdw_size = wdw_size
        self.ref_size = ref_size 
        self.n_feature_maps = n_feature_maps

        self.orientation = orientation

        krn_size = np.array((5, 3))
        img_dims = np.array((ref_size, wdw_size))
        dil_base = np.array((2, 2))

        def minimum_layers(dim_size: int, krn_size: int, dil_base) -> int:
            return int(np.ceil(np.log2((dim_size-1)*(dil_base-1)/(krn_size-1) +1)))
        print([minimum_layers(img_dims[i], krn_size[i], dil_base[i]) for i in range(2)])
        self.n_layers = max([minimum_layers(img_dims[i], krn_size[i], dil_base[i]) for i in range(2)])
        layer_feats = [n_feature_maps]*self.n_layers
     
        layers = []
        for i in range(self.n_layers):
            dil_layer = dil_base ** i
            pad_layer = ((krn_size-1)*dil_layer)
            in_channels = channels if i == 0 else layer_feats[i-1]
            out_channels = layer_feats[i]
            layers.append(TemporalBlock(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=krn_size.tolist(), stride=1, dilation=dil_layer.tolist(), 
                padding=pad_layer, dropout=dropout))
        layers.append(nn.AdaptiveAvgPool2d((1, wdw_size)))
        self.network = nn.Sequential(*layers)

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.ref_size, self.wdw_size))
        print("Input shape: ", x.shape)
        print("Num layers: ", self.n_layers)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)