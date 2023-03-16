# lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch

class LSTMDecoder(LightningModule):

    """ Basic linear sequence. """

    def __init__(self, 
            in_features: int,
            hid_features: int,
            out_features: int,
            hid_layers: int = 1
        ) -> None:

        super().__init__()
        self.save_hyperparameters()
        
        self.in_features = in_features
        
        self.hid_features = hid_features
        self.out_features = out_features

        self.conv = nn.LazyConv2d(out_channels=1, kernel_size=1)
        self.lstm = nn.LSTM(input_size = in_features, hidden_size = hid_features,
                            num_layers = hid_layers, dropout= 0.2, batch_first = True)
        self.linear = nn.Linear(in_features=hid_features, out_features=out_features)

    def forward(self, x):

        out: torch.Tensor = self.conv(x)
        out = out.squeeze()
        out, (hn, cn) = self.lstm(out)

        return self.linear(hn[0])