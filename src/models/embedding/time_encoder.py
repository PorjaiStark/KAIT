import torch
import torch.nn as nn


class TimeEncoder(nn.Module):

    def __init__(self, out_dim=64):
        super().__init__()

        self.gap_dim = 32
        self.month_dim = 2
        self.year_dim = out_dim - self.gap_dim - self.month_dim

        assert self.year_dim > 0

        self.gap_encoder = nn.Linear(
            1,
            self.gap_dim
        )

        self.year_encoder = nn.Linear(
            1,
            self.year_dim
        )


    def forward(self, x):
        """
        Input:
            x: [B,T,4]

            0: gap_norm
            1: month_sin
            2: month_cos
            3: year_norm

        Output:
            [B,T,out_dim]
        """

        gap = x[:,:,0:1]
        month = x[:,:,1:3]
        year = x[:,:,3:4]


        gap_emb = self.gap_encoder(gap)
        year_emb = self.year_encoder(year)


        out = torch.cat(
            [
                gap_emb,
                month,
                year_emb
            ],
            dim=-1
        )

        return out

    
