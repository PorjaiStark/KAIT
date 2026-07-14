import torch
import torch.nn as nn

class WeatherEncoder(nn.Module):

    def __init__(
        self,
        in_dim=30,
        out_dim=128
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim)
        )


    def forward(self, x):
        """
        Input:
            x: [B, T, 30]

        Output:
            [B, T, 128]
        """

        x = self.encoder(x)
        return x

if __name__ == "__main__":

    encoder = WeatherEncoder()
