import math
import torch
import torch.nn as nn


class ContinuousPositionalEncoding(nn.Module):

    def __init__(self, d_model=256, max_period=365.0):
        super().__init__()

        assert d_model % 2 == 0

        self.d_model = d_model

        freq = torch.exp(
            -math.log(max_period)
            * torch.arange(0, d_model, 2).float()
            / d_model
        )

        self.register_buffer("freq", freq)

    def forward(
        self,
        embedding,      # [B,T,256]
        gap_days        # [B,T]
    ):

        pos = gap_days.unsqueeze(-1).float()

        args = pos * self.freq

        pe = torch.zeros_like(embedding)

        pe[...,0::2] = torch.sin(args)
        pe[...,1::2] = torch.cos(args)

        return embedding + pe