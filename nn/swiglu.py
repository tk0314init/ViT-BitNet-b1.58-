import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()
    
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.sigmoid(gate)