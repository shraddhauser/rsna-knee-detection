"""
Model Module: CoPAS Multi-Plane Attention Network
RSNA 2026 Knee Abnormality Detection AI Challenge
Based on Qiu et al. (Nature Communications 2024)
"""

import torch
import torch.nn as nn
from torchvision import models

class CoPASMultiPlaneModel(nn.Module):
    """
    CoPAS (Co-Plane Attention across Sequences) multi-plane architecture.
    Takes 2.5D slabs from Sagittal, Coronal, and Axial planes,
    encodes them via a shared backbone, and fuses them via cross-plane multi-head attention.
    """
    def __init__(self, num_classes: int = 12, num_slices: int = 16):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        orig_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            num_slices, orig_conv.out_channels, 
            kernel_size=orig_conv.kernel_size, 
            stride=orig_conv.stride, 
            padding=orig_conv.padding, 
            bias=False
        )
        self.encoder = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280

        # Cross-Plane Multi-Head Attention
        self.plane_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)

        # Multi-task 12-class prediction head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
        """
        sag, cor, ax shapes: [Batch, num_slices, 224, 224]
        Returns raw logits: [Batch, num_classes]
        """
        f_sag = self.pool(self.encoder(sag)).flatten(1)
        f_cor = self.pool(self.encoder(cor)).flatten(1)
        f_ax = self.pool(self.encoder(ax)).flatten(1)

        planes = torch.stack([f_sag, f_cor, f_ax], dim=1)
        attn_out, _ = self.plane_attn(planes, planes, planes)
        fused = self.norm(planes + attn_out).mean(dim=1)

        return self.classifier(fused)
