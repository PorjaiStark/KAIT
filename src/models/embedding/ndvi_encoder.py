import torch
import torch.nn as nn


class NDVIEncoder(nn.Module):

    def __init__(self, out_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1,32),
            nn.GELU(),
            nn.Linear(32,out_dim),
            nn.LayerNorm(out_dim)
        )


    def forward(self,x):
        """
        x:
        [B,T,1]

        return:
        [B,T,64]
        """

        return self.encoder(x)



if __name__ == "__main__":

    encoder = NDVIEncoder()


    