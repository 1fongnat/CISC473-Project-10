import torch.nn as nn
from .common import conv, deconv
from ..entropy_models import EntropyBottleneck, GaussianConditional

class Hyperprior(nn.Module):
    """
    The Hyperprior model, which learns a prior on the latent representation `y`
    to improve entropy coding. It consists of a hyper-encoder (`h_a`) and a
    hyper-decoder (`h_s`).
    """
    def __init__(self, N=192, M=192): # N for hyper is M from encoder
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        # We use a placeholder for scale_table, it will be updated
        self.gaussian_conditional = GaussianConditional(None) 
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        
        # This will be called to initialize the CDFs for the GaussianConditional
        self.gaussian_conditional.update_scale_table([0.11, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

    def forward(self, y):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        
        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def aux_loss(self):
        """The auxiliary loss to train the entropy bottleneck layer."""
        return self.entropy_bottleneck.loss()