# src/loss.py
import torch.nn as nn
import torch
import math
from .perceptual_loss import PerceptualLoss

class RateDistortionLoss(nn.Module):
    """
    Custom rate-distortion loss, now including a perceptual (LPIPS) term.
    """
    def __init__(self, lmbda=1e-2, lmbda_lpips=5.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.lmbda_lpips = lmbda_lpips
        self.lpips = PerceptualLoss(net='alex') # Initialize the perceptual loss

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # 1. Rate Loss (bits per pixel)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        # 2. Distortion Loss (MSE)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        
        # 3. Perceptual Loss (LPIPS)
        out["lpips_loss"] = self.lpips(output["x_hat"], target).mean()

        # Total Weighted Loss
        out["loss"] = (self.lmbda * 255**2 * out["mse_loss"]) + \
                      out["bpp_loss"] + \
                      (self.lmbda_lpips * out["lpips_loss"])

        return out