import torch
from torch import nn

from nn.bitlinear import BitLinear


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, block_size,):
        super().__init__()
        self.key = BitLinear(in_features=embed_dim, out_features=embed_dim)
        self.query = BitLinear(in_features=embed_dim, out_features=embed_dim)
        self.value = BitLinear(in_features=embed_dim, out_features=embed_dim)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
                        
        N, seq_len, _ = x.shape
        weight = (q @ k.transpose(-2, -1)) * (N ** -0.5)
        
        
        mask = self.tril[:seq_len, :seq_len]
        weight = weight.masked_fill(mask == 0, float('-inf'))

        attntion = torch.softmax(weight, dim=-1)
        output = attntion @ v
        
        return output
        