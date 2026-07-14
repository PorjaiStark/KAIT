import torch
import torch.nn as nn


class NDVIEncoder(nn.Module):

    def __init__(self, out_dim=16):
        super().__init__()

        self.encoder = nn.Linear(1,out_dim)


    def forward(self, x):
        """
        Input:
            x: [B,T,1]

        Output:
            [B,T,16]
        """

        return self.encoder(x)


if __name__ == "__main__":

    encoder = NDVIEncoder()

    