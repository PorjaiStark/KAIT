import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


    def forward(
        self,
        x,
        mask=None
    ):
        """
        x:
            [B,T,256]

        mask:
            [B,T]
            True = valid
            False = padding
        """

        if mask is not None:
            padding_mask = ~mask
        else:
            padding_mask = None


        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        return x
    
if __name__ == "__main__":

    model = TemporalTransformer()

    x = torch.randn(
        4,
        17,
        256
    )

    mask = torch.ones(
        4,
        17,
        dtype=torch.bool
    )

    out = model(
        x,
        mask
    )

    print("input:", x.shape)
    print("output:", out.shape)