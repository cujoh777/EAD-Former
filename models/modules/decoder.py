"""Multi-scale decoder component used by SwinEADFormer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Fuse an upsampled representation with a shallow difference skip."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                3,
                padding=1,
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.conv(torch.cat([x, skip], dim=1))
