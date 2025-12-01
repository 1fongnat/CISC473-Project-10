import torch
import torch.nn as nn
from .common import conv

class FlexTokTokenizer(nn.Module):
    """
    A simplified FlexTok implementation.
    1. Patchifies the image.
    2. Uses a Transformer Encoder with learnable 'Register Tokens'.
    3. The registers become the semantic tokens representing the image.
    """
    def __init__(self, in_channels=3, token_dim=192, num_tokens=16, num_layers=2):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        
        # 1. Patch Embedding (similar to ViT)
        # Reduces 256x256 image -> 16x16 grid of patches
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, token_dim, kernel_size=16, stride=16),
            nn.Flatten(2) # [B, Dim, N_Patches]
        )
        
        # 2. Learnable Register Tokens (The core of FlexTok)
        # These act as 'queries' to gather info from the image patches
        self.registers = nn.Parameter(torch.randn(1, num_tokens, token_dim))
        
        # 3. Position Embeddings for the image patches
        self.num_patches = (256 // 16) ** 2 # Assuming 256x256 input
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, token_dim))
        
        # 4. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, 
            nhead=4, 
            dim_feedforward=token_dim*2, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: [B, 3, 256, 256]
        B = x.size(0)
        
        # 1. Prepare Image Patches
        # [B, Dim, N_Patches] -> [B, N_Patches, Dim]
        patches = self.patch_embed(x).transpose(1, 2)
        patches = patches + self.pos_embed
        
        # 2. Prepare Registers
        # Expand registers for batch size: [B, Num_Tokens, Dim]
        registers = self.registers.expand(B, -1, -1)
        
        # 3. Concatenate: [Registers, Patches]
        # The transformer processes them all together. 
        # The registers will attend to the patches to learn the image content.
        sequence = torch.cat([registers, patches], dim=1)
        
        # 4. Run Transformer
        encoded_sequence = self.transformer(sequence)
        
        # 5. Extract only the updated Registers (the Tokens)
        # We discard the image patches now; the registers hold the summary.
        out_tokens = encoded_sequence[:, :self.num_tokens, :]
        
        return out_tokens