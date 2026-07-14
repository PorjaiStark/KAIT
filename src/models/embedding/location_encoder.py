import torch
import torch.nn as nn

class LocationEncoder(nn.Module):

    def __init__( self, out_dim = 32 ):
        super().__init__()

        self.encoder = nn.Linear( 3, out_dim)

    def forward(self, x):
        """
        Input:
            x:
            [B,T,3]

            x,y,z coordinate

        Output:
            [B,T,32]
        """

        x = self.encoder(x)
        return x



if __name__ == "__main__":

    encoder = LocationEncoder()

    x = torch.randn(
        4,
        19,
        3
    )

    out = encoder(x)

    print("input:", x.shape)
    print("output:", out.shape)