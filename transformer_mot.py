import torch
import torch.nn as nn

from mot import MixtureOfTokensLayer
from transformer_vanilla import VanillaTransformer

N_HEAD = 4
N_LAYERS = 6
D_FF = 512


class MixtureOfTokensDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
            self,
            d_model: int,
            nhead: int = N_HEAD,
            dim_feedforward: int = D_FF,
            **kwargs
    ):
        super(MixtureOfTokensDecoderLayer, self).__init__(d_model, nhead, dim_feedforward=dim_feedforward)
        self.mot = MixtureOfTokensLayer(d_model, dim_feedforward, **kwargs)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mot(x)


class MixtureOfTokens(VanillaTransformer):
    def get_decoder(self) -> nn.TransformerDecoder:
        return nn.TransformerDecoder(
            MixtureOfTokensDecoderLayer(self.d_model, self.nhead, self.d_ff, sparsity_dim=1),
            num_layers=self.num_layers
        )
