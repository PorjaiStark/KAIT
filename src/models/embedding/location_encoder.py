import torch
import torch.nn as nn

class LocationEncoder(nn.Module):

    def __init__(self, in_dim=3, out_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim)
        )


    def forward(self, x):
        """
        Input:
            x:
            [B,T,3]

            x,y,z coordinate

        Output:
            [B,T,64]
        """

        return self.encoder(x)



if __name__ == "__main__":

    encoder = LocationEncoder()

