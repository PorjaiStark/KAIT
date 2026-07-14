import torch
import torch.nn as nn


class ConcatFusion(nn.Module):

    def __init__( self, input_dim=370, d_model=256):
        super().__init__()

        self.projection = nn.Linear( input_dim, d_model)


    def forward(
        self,
        ndvi,
        time,
        location,
        weather,
        sentinel
    ):
        """
        Inputs:

        ndvi:
            [B,T,16]

        time:
            [B,T,66]

        location:
            [B,T,32]

        weather:
            [B,T,128]

        sentinel:
            [B,T,128]


        Output:

            [B,T,128]
        """

        x = torch.cat([ ndvi, time, location, weather, sentinel], dim=-1)
        x = self.projection(x)

        return x
    