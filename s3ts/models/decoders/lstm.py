# lightning
from pytorch_lightning import LightningModule
import torch.nn as nn

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

        self.lstm = nn.LSTM(input_size = in_features, hidden_size = hid_features,
                            num_layers = hid_layers, dropout= 0.2, batch_first = True)
        self.linear = nn.Linear(in_features=hid_layers, out_features=out_features)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.linear(hn[0]).flatten() 