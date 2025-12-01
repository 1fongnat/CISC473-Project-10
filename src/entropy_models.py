import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from .network.common import LowerBound

def _pmf_to_quantized_cdf(pmf: List[float], precision: int) -> List[int]:
    """Helper function to create a quantized CDF from a PMF."""
    cdf = [0] * (len(pmf) + 1)
    
    for i in range(len(pmf)):
        cdf[i+1] = cdf[i] + int(round(pmf[i] * (1 << precision)))
    
    total = cdf[-1]
    if total == 0:
        for i in range(1, len(cdf)):
            cdf[i] = (i * (1 << precision)) // len(pmf)
        cdf[-1] = 1 << precision
        return cdf
        
    for i in range(1, len(cdf)):
        cdf[i] = (cdf[i] * (1 << precision)) // total
    
    for i in range(len(cdf) - 1):
        if cdf[i] >= cdf[i+1]:
            best_j = -1
            max_freq = 0
            for j in range(len(cdf) - 1):
                freq = cdf[j+1] - cdf[j]
                if freq > 1 and freq > max_freq:
                    max_freq = freq
                    best_j = j
            
            if best_j != -1:
                for k in range(best_j + 1, len(cdf)):
                    cdf[k] -= 1
                cdf[i+1] = cdf[i] + 1
            else:
                 cdf[i+1] = cdf[i] + 1

    cdf[-1] = 1 << precision
    return cdf


class EntropyModel(nn.Module):
    def __init__(self, likelihood_bound: float = 1e-9):
        super().__init__()
        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def quantize(self, inputs: Tensor, mode: str, means: Optional[Tensor] = None) -> Tensor:
        if mode == "noise":
            noise = torch.empty_like(inputs).uniform_(-0.5, 0.5)
            return inputs + noise

        outputs = inputs.clone()
        if means is not None:
            outputs -= means
        outputs = torch.round(outputs)
        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs
        
        return outputs.int()

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = torch.tensor(_pmf_to_quantized_cdf(prob.tolist(), 16))
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

class EntropyBottleneck(EntropyModel):
    def __init__(self, channels: int, tail_mass: float = 1e-9, init_scale: float = 10.0):
        super().__init__()
        self.channels = int(channels)
        self.tail_mass = float(tail_mass)
        self.init_scale = float(init_scale)
        
        filters = (3, 3, 3)
        self.filters = tuple(int(f) for f in filters)
        
        # Create parameters
        filters_ = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters_[i + 1]))
            matrix = torch.Tensor(channels, filters_[i + 1], filters_[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters_[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters_[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))


    def _get_medians(self) -> Tensor:
        return self.quantiles[:, :, 1:2].detach()
    
    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits
        
    def forward(self, x: Tensor, training: Optional[bool] = None) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        
        #
        perm = np.arange(len(x.shape))
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)]

        x_perm = x.permute(*perm).contiguous()
        shape = x_perm.size()
        values = x_perm.reshape(x_perm.size(0), 1, -1)
        
        outputs = self.quantize(values, "noise" if training else "dequantize", self._get_medians())

        half = float(0.5)
        v0 = outputs - half
        v1 = outputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower + upper).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood
        
    def loss(self) -> Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=False)
        return torch.abs(logits - self.target).sum()

class GaussianConditional(EntropyModel):
    def __init__(self, scale_table: Optional[Union[List, Tuple]], scale_bound: float = 0.11, tail_mass: float = 1e-9):
        super().__init__()
        self.lower_bound_scale = LowerBound(scale_bound) if scale_bound > 0 else None
        self.tail_mass = float(tail_mass)
        if scale_table:
            self.register_buffer("scale_table", self._prepare_scale_table(scale_table))

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        return 0.5 * torch.erfc(-(2**-0.5) * inputs)
        
    def _likelihood(self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None) -> Tensor:
        values = inputs
        if means is not None:
            values = inputs - means
            
        if self.lower_bound_scale:
            scales = self.lower_bound_scale(scales)
        
        values = torch.abs(values)
        upper = self._standardized_cumulative((0.5 - values) / scales)
        lower = self._standardized_cumulative((-0.5 - values) / scales)
        likelihood = upper - lower
        return likelihood
        
    def forward(self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None, training: Optional[bool] = None) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
            
        outputs = self.quantize(inputs, "noise" if training else "dequantize", means)
        likelihood = self._likelihood(outputs, scales, means)
        
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
            
        return outputs, likelihood
        
    def update_scale_table(self, scale_table, force=False):
        if hasattr(self, '_offset') and self._offset.numel() > 0 and not force:
            return False
        
        device = self.lower_bound_scale.bound.device if self.lower_bound_scale else 'cpu'
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        
        # Re-calculate CDFs
        multiplier = -scipy.stats.norm.ppf(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        samples = torch.abs(torch.arange(max_length, device=device).int() - pmf_center[:, None])
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower
        tail_mass = 2 * lower[:, :1]
        
        self._quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2
        return True