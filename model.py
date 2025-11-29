import torch.nn as nn
import torch.optim as optim
from .network.encoder import Encoder
from .network.generator import Generator
from .network.hyperprior import Hyperprior

class ImageCompressionModel(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.encoder = Encoder(N=N, M=M)
        self.generator = Generator(out_channels=3, N=N, M=M)
        self.hyperprior = Hyperprior(N=M, M=M)

    def forward(self, x):
        y = self.encoder(x)
        y_hat, likelihoods = self.hyperprior(y)
        x_hat = self.generator(y_hat)
        return {"x_hat": x_hat.clamp(0, 1), "likelihoods": likelihoods}

    def aux_loss(self):
        return self.hyperprior.aux_loss()

    def configure_optimizers(self, args):
        params = {n for n, p in self.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
        aux_params = {n for n, p in self.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
        p_dict = dict(self.named_parameters())
        return optim.Adam((p_dict[n] for n in sorted(params)), lr=args.learning_rate), \
               optim.Adam((p_dict[n] for n in sorted(aux_params)), lr=args.aux_learning_rate)