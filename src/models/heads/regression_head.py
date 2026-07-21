import torch
import torch.nn as nn


class NDVIRegressionHead(nn.Module):
    """
    Regression decoder for NDVI prediction.

        h = ReLU(Linear1(H))
        y = Linear2(Dropout(h))


    Input:
        [B,T,d_model]


    Output:
        [B,T,1]

    """


    def __init__(
        self,
        d_model=256,
        dropout=0.1
    ):
        super().__init__()


        self.decoder = nn.Sequential(

            nn.Linear(
                d_model,
                d_model // 2
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model // 2,
                1
            )
        )


    def forward(
        self,
        x
    ):
        """
        Args:

            x:
                Transformer output

                [B,T_future,256]


        Returns:

            NDVI prediction

                [B,T_future,1]

        """

        return self.decoder(
            x
        )



if __name__ == "__main__":


    head = NDVIRegressionHead(
        d_model=256,
        dropout=0.1
    )


