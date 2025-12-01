"""
FILE: src/loss.py
PURPOSE: Defines the Rate-Distortion Loss Function with Saliency Weighting.
"""

import torch.nn as nn
import torch
import math
import numpy as np
from .perceptual_loss import PerceptualLoss

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, lmbda_lpips=10.0, use_saliency=False):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none') # Important: 'none' to keep pixel-wise errors
        self.lmbda = lmbda
        self.lmbda_lpips = lmbda_lpips
        self.use_saliency = use_saliency
        self.lpips = PerceptualLoss(net='alex') 

    def get_center_saliency_map(self, N, H, W, device):
        """
        Creates a Gaussian mask centered on the image.
        Center pixels = High weight (1.0)
        Corner pixels = Low weight (e.g., 0.2)
        """
        # Create a grid of coordinates
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Calculate distance from center (0,0)
        dist = torch.sqrt(xx**2 + yy**2)
        
        # Gaussian function: e^(-dist^2 / sigma)
        # Sigma controls how big the "focus area" is.
        sigma = 0.5
        mask = torch.exp(-dist**2 / (2 * sigma**2))
        
        # Normalize mask to be between 0.2 (background) and 1.0 (center)
        mask = 0.2 + 0.8 * mask 
        
        return mask.to(device).unsqueeze(0).unsqueeze(0).expand(N, 3, H, W)

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # 1. Rate Loss (Bitrate)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        # 2. Distortion Loss (MSE)
        # Calculate pixel-wise squared error
        pixel_mse = (output["x_hat"] - target) ** 2
        
        if self.use_saliency:
            # [UNIQUE FEATURE]: Saliency Weighting
            # Generate a weight map (Center Bias)
            weight_map = self.get_center_saliency_map(N, H, W, target.device)
            
            # Multiply error by importance. 
            # Errors in the center are penalized heavily. Errors in corners are ignored.
            weighted_mse = pixel_mse * weight_map
            out["mse_loss"] = weighted_mse.mean()
        else:
            # Standard MSE
            out["mse_loss"] = pixel_mse.mean()
        
        # 3. Perceptual Loss (LPIPS)
        if self.lmbda_lpips > 0:
            out["lpips_loss"] = self.lpips(output["x_hat"], target).mean()
        else:
            out["lpips_loss"] = torch.tensor(0.0)

        # Total Loss
        out["loss"] = (self.lmbda * 255**2 * out["mse_loss"]) + \
                      out["bpp_loss"] + \
                      (self.lmbda_lpips * out["lpips_loss"])

        return out