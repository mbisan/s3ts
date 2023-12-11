import torch

class SimpleCNN(torch.nn.Module):
    
    def __init__(self, channels=1, ref_size=32, 
            wdw_size=32, n_feature_maps=32):
        super().__init__()
        
        self.channels = channels
        self.ref_size = ref_size
        self.wdw_size = wdw_size
        self.n_feature_maps = n_feature_maps

        layer1_input = ref_size%2
        layer2_input = (layer1_input-3)//2
        layer3_input = (layer2_input-3)//2

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels, out_channels=n_feature_maps//2, kernel_size=5 if layer1_input%2==0 else 4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=n_feature_maps//2, out_channels=n_feature_maps, kernel_size=5 if layer2_input%2==0 else 4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=n_feature_maps, out_channels=n_feature_maps*2, kernel_size=5 if layer3_input%2==0 else 4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.ref_size, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape
