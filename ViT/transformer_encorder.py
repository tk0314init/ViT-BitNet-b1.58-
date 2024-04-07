import torch
from torch import nn

from ViT.selfattention import SelfAttention
from nn.bitlinear import BitLinear
from nn.rsmnorm import RMSNorm
from nn.swiglu import SwiGLU


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.attention = SelfAttention(embed_dim, self.head_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        N, seq_length, embed_dim = x.shape
        x = x.reshape(N, seq_length, self.num_heads, self.head_dim)
        
        attn_out = self.attention(x)
        attn_out = attn_out.reshape(N, seq_length, embed_dim)
        
        out = self.linear(attn_out)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, block_size):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_dim, block_size)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        
        self.linear1 = BitLinear(embed_dim, forward_expansion * embed_dim * 2)
        self.swiglu = SwiGLU()
        self.linear2 = BitLinear(forward_expansion * embed_dim, embed_dim)
  
        
    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        forward_out = self.linear1(x)
        forward_out = self.swiglu(forward_out)
        forward_out = self.linear2(forward_out)
        out = self.norm2(x + forward_out)
        return out



class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, forward_expansion, block_size):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, forward_expansion, block_size)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x