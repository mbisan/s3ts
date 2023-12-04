import torch

from s3ts.dtw_layer.dtw import torch_dtw

class DTWLayer(torch.nn.Module):
    def __init__(self, n_patts, d_patts, l_patts, l_out: int = None, rho: float = 1) -> None:
        super().__init__()

        if not l_out is None:
            self.l_out = l_patts
        else:
            self.l_out = l_patts

        self.w: torch.float32 = rho ** (1/l_patts)
        self.patts = torch.nn.Parameter(torch.randn(n_patts, d_patts, l_patts))
    
    def forward(self, x):
        return torch_dtw.apply(x, self.patts, self.w)[0][:,:,:,-self.l_out:]