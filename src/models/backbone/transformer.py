import torch
import torch.nn as nn


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer Encoder for spatio-temporal representation learning.

    Follow:
        BERT-style Transformer Encoder

    Each block:
        Multi-head self attention
        Residual connection
        LayerNorm
        Feed-forward network
        Residual connection
        LayerNorm


    Input:
        x:
            [B,T,d_model]


    Output:
        [B,T,d_model]

    """


    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False
        )


        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)


    def forward(
        self,
        x,
        padding_mask=None
    ):
        """
        Args:

            x:
                Transformer input

                [B,T,256]


            padding_mask:

                True = ignore token

                [B,T]


        Returns:

            encoded representation

                [B,T,256]

        """


        x = self.transformer(x,src_key_padding_mask=padding_mask)
        x = self.final_norm(x)

        return x



if __name__ == "__main__":


    model = TemporalTransformerEncoder(
        d_model=256,
        nhead=8,
        num_layers=4
    )