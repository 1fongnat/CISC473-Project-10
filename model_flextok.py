import torch
import torch.nn as nn
import torch.optim as optim
from .network.encoder import Encoder
from .network.hyperprior import Hyperprior
from .network.tokenizer import FlexTokTokenizer
from .network.common import deconv
# We will define the FlexTok-specific Generator directly here to be safe
from .network.attention import CrossAttentionBlock 

class FlexTokGenerator(nn.Module):
    def __init__(self, out_channels=3, N=128, M=192):
        super().__init__()
        # Standard Decoder Layers split for injection
        self.layer1 = nn.Sequential(deconv(M, N), nn.GELU())
        self.layer2 = nn.Sequential(deconv(N, N), nn.GELU())
        
        # The Attention Block
        self.flextok_attn = CrossAttentionBlock(spatial_channels=N, token_dim=M)

        self.layer3 = nn.Sequential(deconv(N, N), nn.GELU())
        self.layer4 = deconv(N, out_channels)

    def forward(self, x, tokens):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flextok_attn(x, tokens) # Always apply attention
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class FlexTokImageCompressionModel(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.encoder = Encoder(N=N, M=M)
        self.hyperprior = Hyperprior(N=M, M=M)
        self.tokenizer = FlexTokTokenizer(in_channels=3, token_dim=M, num_tokens=16)
        self.generator = FlexTokGenerator(out_channels=3, N=N, M=M)

    def forward(self, x):
        # 1. Tokenize
        tokens = self.tokenizer(x)
        
        # 2. Compress Latents
        y = self.encoder(x)
        y_hat, likelihoods = self.hyperprior(y)
        
        # 3. Generate with Guidance
        x_hat = self.generator(y_hat, tokens)
        
        return {
            "x_hat": x_hat.clamp(0, 1),
            "likelihoods": likelihoods,
            "tokens": tokens
        }

    def aux_loss(self):
        return self.hyperprior.aux_loss()

    def configure_optimizers(self, args):
        # Standard optimization
        params = {n for n, p in self.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
        aux_params = {n for n, p in self.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
        p_dict = dict(self.named_parameters())
        optimizer = optim.Adam((p_dict[n] for n in sorted(params)), lr=args.learning_rate)
        aux_optimizer = optim.Adam((p_dict[n] for n in sorted(aux_params)), lr=args.aux_learning_rate)
        return optimizer, aux_optimizer