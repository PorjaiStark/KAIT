import torch
import torch.nn as nn


class MultiModalFusion(nn.Module):
    """
    Concatenate multimodal embeddings
    and project into Transformer dimension.

    Input:

        NDVI:
            [B,T,64]

        Time:
            [B,T,64]

        Location:
            [B,T,32]

        Weather:
            [B,T,128]

        Sentinel:
            [B,T,128]


    Concatenation:

        64 + 64 + 32 + 128 + 128 = 416


    Output:

        [B,T,d_model]

    """


    def __init__(
        self,
        d_model=256
    ):
        super().__init__()


        self.input_dim = (
            64 +     # NDVI
            64 +     # Time
            32 +     # Location
            128 +    # Weather
            128      # Sentinel
        )


        self.projection = nn.Sequential(

            nn.Linear(
                self.input_dim,
                d_model
            ),

            nn.LayerNorm(
                d_model
            )
        )


    def pad_sequence(
        self,
        x,
        target_len
    ):
        """
        Pad weather/sentinel future steps.

        Input:
            [B,Obs,D]

        Output:
            [B,T,D]

        """

        B,L,D = x.shape


        if L == target_len:
            return x


        pad_len = target_len - L


        padding = torch.zeros(
            B,
            pad_len,
            D,
            device=x.device,
            dtype=x.dtype
        )


        return torch.cat(
            [
                x,
                padding
            ],
            dim=1
        )


    def forward(
        self,
        ndvi,
        time,
        location,
        weather,
        sentinel
    ):

        """
        Args:

            ndvi:
                [B,T,64]

            time:
                [B,T,64]

            location:
                [B,T,32]

            weather:
                [B,Obs,128]

            sentinel:
                [B,Obs,128]


        Returns:

            fused:
                [B,T,256]

        """


        B,T,_ = ndvi.shape


        # weather/sentinel only exist in observation
        weather = self.pad_sequence(
            weather,
            T
        )


        sentinel = self.pad_sequence(
            sentinel,
            T
        )


        x = torch.cat(
            [
                ndvi,
                time,
                location,
                weather,
                sentinel
            ],
            dim=-1
        )


        # [B,T,416]
        x = self.projection(
            x
        )


        # [B,T,256]
        return x



if __name__ == "__main__":


    fusion = MultiModalFusion(
        d_model=256
    )
