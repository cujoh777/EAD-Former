"""Architecture-only implementation of the full SwinEADFormer model."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DecoderBlock, DynamicEdgeRouter, EADBlock


class SwinEADFormer(nn.Module):
    """SwinEADFormer for binary building change detection."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=256,
        )

        self.diff_fusion = nn.Sequential(
            nn.Conv2d(384 * 4, 384, 1),
            nn.GroupNorm(8, 384),
            nn.ReLU(inplace=True),
        )

        self.router = DynamicEdgeRouter(384, 768)
        self.ead_block_1 = EADBlock(384)
        self.ead_block_2 = EADBlock(384)

        self.phi_fusion = nn.Sequential(
            nn.Conv2d(384 * 3, 384, 1),
            nn.GroupNorm(8, 384),
            nn.ReLU(inplace=True),
        )

        self.decoder_s2 = DecoderBlock(384, 192, 192)
        self.decoder_s1 = DecoderBlock(192, 96, 96)
        self.main_head = nn.Conv2d(96, 1, 1)
        self.aux_head = nn.Conv2d(384, 1, 1)

    def encode(self, image: torch.Tensor):
        features = self.backbone(image)
        return [
            feature.permute(0, 3, 1, 2).contiguous()
            for feature in features
        ]

    def interact(
        self,
        query_feat: torch.Tensor,
        reference_feat: torch.Tensor,
        router: torch.Tensor,
    ) -> torch.Tensor:
        output = self.ead_block_1(query_feat, reference_feat, router)
        return self.ead_block_2(output, reference_feat, router)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor):
        t1_s1, t1_s2, t1_s3, t1_s4 = self.encode(t1)
        t2_s1, t2_s2, t2_s3, t2_s4 = self.encode(t2)

        learned_difference = self.diff_fusion(
            torch.cat(
                [
                    t1_s3,
                    t2_s3,
                    torch.abs(t1_s3 - t2_s3),
                    t1_s3 * t2_s3,
                ],
                dim=1,
            )
        )

        semantic_difference = torch.abs(t1_s4 - t2_s4)
        router, sparsity_loss = self.router(
            learned_difference,
            semantic_difference,
        )

        output_12 = self.interact(t1_s3, t2_s3, router)
        output_21 = self.interact(t2_s3, t1_s3, router)
        interaction_difference = torch.abs(output_12 - output_21)

        fused_change = self.phi_fusion(
            torch.cat(
                [
                    interaction_difference,
                    learned_difference,
                    torch.abs(t1_s3 - t2_s3),
                ],
                dim=1,
            )
        )

        auxiliary_logits = self.aux_head(fused_change)
        decoded_s2 = self.decoder_s2(
            fused_change,
            torch.abs(t1_s2 - t2_s2),
        )
        decoded_s1 = self.decoder_s1(
            decoded_s2,
            torch.abs(t1_s1 - t2_s1),
        )
        prediction_logits = self.main_head(decoded_s1)

        prediction_logits = F.interpolate(
            prediction_logits,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        auxiliary_logits = F.interpolate(
            auxiliary_logits,
            size=prediction_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        router = F.interpolate(
            router,
            size=prediction_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": prediction_logits,
            "aux": auxiliary_logits,
            "edge": router,
            "sparsity_loss": sparsity_loss,
        }
