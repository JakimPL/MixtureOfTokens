from typing import Optional

import torch
import torch.nn as nn

N_HEAD = 4
N_LAYERS = 6
D_FF = 512


class VanillaTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            nhead: int = N_HEAD,
            d_ff: int = D_FF,
            num_layers: int = N_LAYERS
    ):
        super(VanillaTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff = d_ff
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_decoder = self.get_decoder()

        self.fc = nn.Linear(d_model, vocab_size)

    def get_decoder(self) -> nn.TransformerDecoder:
        return nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.d_model, self.nhead, self.d_ff),
            num_layers=self.num_layers
        )

    @staticmethod
    def generate_square_subsequent_mask(size: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)

        if mask is None:
            mask = self.generate_square_subsequent_mask(x.size(0)).to(x.device)

        memory = torch.zeros_like(x)
        output = self.transformer_decoder(x, memory, tgt_mask=mask)
        output = self.fc(output)
        return output
