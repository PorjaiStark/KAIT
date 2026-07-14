import torch
import torch.nn as nn

from torchgeo.models import resnet18, ResNet18_Weights
from torchvision import models


class SentinelEncoder(nn.Module):

    def __init__(self, out_dim=128,):
        super().__init__()

        # Base ResNet18 architecture
        self.backbone = models.resnet18(weights=None)

        # Sentinel-2 input = 10 bands
        self.backbone.conv1 = nn.Conv2d(
            in_channels=10,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Remove ImageNet classifier
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Projection head
        self.projector = nn.Linear( feature_dim, out_dim)

        # Load TorchGeo Sentinel-2 pretrained weights
        weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
        pretrained = resnet18( weights=weights)
        state_dict = pretrained.state_dict()

        # Remove weights that cannot match
        # conv1 because channel changed from 13 -> 10
        # fc because classifier removed
        del state_dict["conv1.weight"]
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]

        msg = self.backbone.load_state_dict( state_dict, strict=False)
        
        # Keep:
        pretrained_conv = pretrained.state_dict()["conv1.weight"]
        mapping = [
            1,   # B2
            2,   # B3
            3,   # B4
            4,   # B5
            5,   # B6
            6,   # B7
            7,   # B8
            8,   # B8A
            11,  # B11
            12   # B12
        ]

        with torch.no_grad():
            self.backbone.conv1.weight.copy_(
                pretrained_conv[:, mapping, :, :]
            )


    


    def forward(self, x):

        """
        Input:
            x = [B,T,10,H,W]

        Output:
            [B,T,128]
        """

        B,T,C,H,W = x.shape

        x = x.reshape(
            B*T,
            C,
            H,
            W
        )

        # spatial feature extraction
        x = self.backbone(x)
        x = self.projector(x) # project feature dimension
        
        # restore temporal dimension
        x = x.reshape( B, T, -1)
        return x

if __name__ == "__main__":

    encoder = SentinelEncoder()

    