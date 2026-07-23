"""Edge-region-semantic routing module used by SwinEADFormer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicEdgeRouter(nn.Module):
    """Generate an image-pair-specific edge-region-semantic soft gate."""

    def __init__(
        self,
        in_channels: int,
        context_channels: int,
        init_bias: float = -1.0,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(8, in_channels)
        self.to_gray = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        with torch.no_grad():
            self.conv_x.weight.copy_(sobel_x)
            self.conv_y.weight.copy_(sobel_y)
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

        hidden_context = max(context_channels // 4, 8)
        hidden_region = max(in_channels // 4, 8)

        self.semantic_proj = nn.Sequential(
            nn.Conv2d(context_channels, hidden_context, 3, padding=1),
            nn.GroupNorm(8, hidden_context),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_context, 1, 1),
        )
        self.region_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_region, 3, padding=1),
            nn.GroupNorm(8, hidden_region),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_region, 1, 1),
        )
        self.router_bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(
        self,
        diff_feat: torch.Tensor,
        context_feat: torch.Tensor,
    ):
        gray = self.to_gray(self.norm(diff_feat))

        gx = self.conv_x(gray.float())
        gy = self.conv_y(gray.float())
        magnitude = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        magnitude = magnitude / (
            magnitude.amax(dim=(2, 3), keepdim=True) + 1e-6
        )
        magnitude = magnitude.clamp(0, 1).to(dtype=diff_feat.dtype)

        if context_feat.shape[-2:] != diff_feat.shape[-2:]:
            context_feat = F.interpolate(
                context_feat,
                size=diff_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        edge_logit = 2.0 * magnitude - 1.0
        region_logit = self.region_proj(diff_feat)
        semantic_logit = self.semantic_proj(context_feat)
        router = torch.sigmoid(
            edge_logit + region_logit + semantic_logit + self.router_bias
        )
        return router, router.mean()
