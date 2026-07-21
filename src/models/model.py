import torch
import torch.nn as nn

from src.models.fusion.concat_fusion import MultiModalFusion
from src.models.backbone.positional_encoder import ContinuousPositionalEncoding
from src.models.backbone.transformer import TemporalTransformerEncoder
from src.models.heads.regression_head import NDVIRegressionHead


class NDVITransformerModel(nn.Module):

    def __init__(
        self,
        d_model=256
    ):
        super().__init__()


        self.fusion = MultiModalFusion(
            d_model=d_model
        )


        self.pos_encoder = ContinuousPositionalEncoding(
            d_model=d_model
        )


        self.transformer = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1
        )


        self.regression = NDVIRegressionHead(
            d_model=d_model,
            dropout=0.1
        )


    def extract_future_tokens(
        self,
        x,
        future_mask
    ):
        """
        Extract future query tokens and pad.

        Input:
            x:
                [B,T,D]

            future_mask:
                [B,T]


        Output:
            [B,T_future,D]

        """


        B,T,D = x.shape


        future_lengths = future_mask.sum(
            dim=1
        )


        max_future = future_lengths.max()


        outputs = []


        for i in range(B):

            tokens = x[
                i,
                future_mask[i]
            ]


            pad_len = (
                max_future
                -
                tokens.shape[0]
            )


            if pad_len > 0:

                padding = torch.zeros(
                    pad_len,
                    D,
                    device=x.device,
                    dtype=x.dtype
                )


                tokens = torch.cat(
                    [
                        tokens,
                        padding
                    ],
                    dim=0
                )


            outputs.append(
                tokens
            )


        return torch.stack(
            outputs,
            dim=0
        )



    def forward(
        self,
        ndvi,
        time,
        location,
        weather,
        sentinel,
        gap_norm,
        future_query_mask,
        padding_mask=None
    ):

        # Fusion
        x = self.fusion(
            ndvi,
            time,
            location,
            weather,
            sentinel
        )


        # [B,T,256]

        # Continuous PE
        x = self.pos_encoder(
            x,
            gap_norm
        )
        
        pre_transformer = x.clone()
        
        # Transformer
        if padding_mask is not None:
            padding_mask = ~padding_mask
  

        x = self.transformer(
            x,
            padding_mask
            )
        
        transformer_output = x.clone()
        
        # Future query tokens
        x_future = self.extract_future_tokens(
            x,
            future_query_mask
        )
        future_embedding = x_future
         
        # [B,T_future,256]

        # Regression
        prediction = self.regression(
            future_embedding
        )
        
        return {
            "prediction": prediction,
            "pre_transformer": pre_transformer,
            "transformer_output": transformer_output,
            "future_embedding": future_embedding,
        }