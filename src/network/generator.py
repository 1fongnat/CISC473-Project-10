import torch.nn as nn
from .common import deconv

class Generator(nn.Module):
    def __init__(self, out_channels=3, N=128, M=192):
        super().__init__()
        self.g_s = nn.Sequential(
            deconv(M, N), nn.GELU(),
            deconv(N, N), nn.GELU(),
            deconv(N, N), nn.GELU(),
            deconv(N, out_channels)
        )

    def forward(self, x):
        return self.g_s(x)