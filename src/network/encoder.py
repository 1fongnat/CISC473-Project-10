import torch.nn as nn
from .common import conv

class Encoder(nn.Module):
    """
    The Encoder part of the autoencoder. It compresses the input image into a latent representation.
    """
    def __init__(self, in_channels=3, N=128, M=192):
        super().__init__()
        self.g_a = nn.Sequential(
            conv(in_channels, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, M, kernel_size=5, stride=2),
        )

    def forward(self, x):
        return self.g_a(x)