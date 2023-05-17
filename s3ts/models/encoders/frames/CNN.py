# lightning
from pytorch_lightning import LightningModule

import torch.nn as nn
import torch

class CNN_DF(LightningModule):

    """ Basic CNN for dissimilarity frames. """

    def __init__(self, channels: int, wdw_size: int, ref_size: int,
                 n_feature_maps: int = 32):
        super().__init__()

        # register parameters
        self.channels = channels
        self.wdw_size = wdw_size
        self.ref_size = ref_size
        self.n_feature_maps = n_feature_maps

        # convolutional layer 0
        self.cnn_0 = nn.Sequential(nn.Conv2d(in_channels=channels, 
            out_channels=self.n_feature_maps//2, kernel_size=3, padding='same'),
            nn.ReLU(), nn.MaxPool2d(kernel_size=(4,1)))
        
        # convolutional layer 1
        self.cnn_1 = nn.Sequential(nn.Conv2d(in_channels=self.n_feature_maps//2, 
            out_channels=self.n_feature_maps, kernel_size=3, padding='same'),
            nn.ReLU(), nn.AvgPool2d(kernel_size=(4,1)), nn.Dropout(0.35))
        
        # convolutional layer 2
        self.cnn_2 = nn.Sequential(nn.Conv2d(in_channels=self.n_feature_maps, 
            out_channels=self.n_feature_maps*2, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=self.n_feature_maps*2),
            nn.ReLU(), nn.Dropout(0.4))

        # convolutional layer 3
        self.cnn_3 = nn.Conv2d(in_channels=self.n_feature_maps*2, 
            out_channels=self.n_feature_maps*2, kernel_size=3, padding='same')

        #self.linear_1 = self.dynamic_linear((1, channels, ref_size, window_size)))
        #self.linear_2 = nn.Linear(in_features=self.n_feature_maps * 4, out_features=self.n_feature_maps * 8)

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.ref_size, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape

    # def dynamic_linear(self, image_dim):
    #     x = torch.rand(*(image_dim))
    #     features = self.model(x.float())
    #     flat = features.view(features.size(0), -1)
    #     return nn.Linear(in_features=flat.size(1), out_features=self.n_feature_maps * 4)

    def forward(self, x):
        feats = self.cnn_0(x.float())
        feats = self.cnn_1(feats)
        feats = self.cnn_2(feats)
        feats = self.cnn_3(feats)      
        #flat = features.view(features.size(0), -1)
        #lin_1 = self.linear_1(flat)
        #return self.linear_2(lin_1)
        return feats