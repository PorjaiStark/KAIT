import torch
import torch.nn as nn


class TimeEncoder(nn.Module):

    def __init__(
        self,
        gap_dim=32,
        year_dim=32
    ):
        super().__init__()
        self.gap_encoder = nn.Linear(1, gap_dim)
        self.year_encoder = nn.Linear(1,year_dim)


    def forward(self, x):
        """
        Input:
            x:
            [B,T,4]

            x[:,:,0] = gap_norm
            x[:,:,1] = month_sin
            x[:,:,2] = month_cos
            x[:,:,3] = year_norm


        Output:
            [B,T,66]

        """

        gap = x[:,:,0:1]
        month = x[:,:,1:3]
        year = x[:,:,3:4]


        gap_emb = self.gap_encoder(gap)
        year_emb = self.year_encoder(year)


        out = torch.cat([ gap_emb, month, year_emb], dim=-1)
        return out

if __name__ == "__main__":
    encoder = TimeEncoder()
