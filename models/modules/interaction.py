"""Router-gated bidirectional interaction components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterGatedCrossAttention(nn.Module):
    """Cross-temporal attention with output-level soft spatial gating."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop

    def forward(
        self,
        query_feat: torch.Tensor,
        reference_feat: torch.Tensor,
        router: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = query_feat.shape
        num_tokens = height * width

        query_tokens = query_feat.flatten(2).transpose(1, 2)
        reference_tokens = reference_feat.flatten(2).transpose(1, 2)
        spatial_gate = router.flatten(2).transpose(1, 2)

        query = self.q(query_tokens)
        key, value = self.kv(reference_tokens).chunk(2, dim=-1)

        def reshape_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(
                batch,
                num_tokens,
                self.num_heads,
                channels // self.num_heads,
            ).transpose(1, 2)

        query = reshape_heads(query)
        key = reshape_heads(key)
        value = reshape_heads(value)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, num_tokens, channels)
        output = self.proj_drop(self.proj(output))
        output = output * spatial_gate
        return output.transpose(1, 2).view(batch, channels, height, width)


class EADBlock(nn.Module):
    """Edge-aware dynamic cross-temporal interaction block."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm_query = nn.GroupNorm(8, dim)
        self.norm_reference = nn.GroupNorm(8, dim)
        self.norm_mlp = nn.GroupNorm(8, dim)
        self.attention = RouterGatedCrossAttention(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1),
        )

    def forward(
        self,
        query_feat: torch.Tensor,
        reference_feat: torch.Tensor,
        router: torch.Tensor,
    ) -> torch.Tensor:
        interaction = self.attention(
            self.norm_query(query_feat),
            self.norm_reference(reference_feat),
            router,
        )
        output = query_feat + interaction
        return output + 0.5 * self.mlp(self.norm_mlp(output))
