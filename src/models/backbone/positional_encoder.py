import math
import torch
import torch.nn as nn


class ContinuousPositionalEncoding(nn.Module):
    """
    Continuous Temporal Fourier Encoding.

    Instead of using discrete token index:
        t = 0,1,2,...

    We use continuous temporal position:
        gap_norm = gap_days / observe_window_days 90

    Input:
        embedding:
            [B,T,d_model]

        gap_norm:
            [B,T]


    Output:
        [B,T,d_model]
    """


    def __init__(
        self,
        d_model=256
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        freq = torch.arange( 0, d_model // 2).float()
        self.register_buffer("freq", freq)


    def forward(
        self,
        embedding,
        gap_norm
    ):
        """
        Args:

            embedding:
                fused representation
                [B,T,d_model]


            gap_norm:
                continuous temporal position
                [B,T]

        """

        t = gap_norm.unsqueeze(-1).float()
        args = (2 *math.pi * t * self.freq)
        pe = torch.zeros_like(embedding)
        pe[...,0::2] = torch.sin(args)
        pe[...,1::2] = torch.cos(args)

        return embedding + pe



if __name__ == "__main__":

    encoder = ContinuousPositionalEncoding(d_model=256)

