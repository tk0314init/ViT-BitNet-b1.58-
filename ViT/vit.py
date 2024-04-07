import torch
from torch import nn

from nn.bitlinear import BitLinear
from ViT.patchembedding import PatchEmbedding
from ViT.transformer_encorder import TransformerEncoder



class ViT(nn.Module):
    def __init__(self, num_patches, num_classes, patch_size, embed_dim, num_layers, num_heads, dropout, in_channels, forward_expansion, block_size):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        
        self.encoder_block = TransformerEncoder(embed_dim, num_heads, num_layers, forward_expansion, block_size)

        self.mlp_head = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_block(x)
        x = self.mlp_head(x[:, 0, :])
        return x