import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-4):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, _input):
        rms = torch.sqrt(torch.mean(_input ** 2, dim=-1, keepdim=True) + self.eps)
        normalized_input = _input / rms * self.scale
        return normalized_input