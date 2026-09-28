"""
Loss Functions Module: Combined Focal + BCE Loss
RSNA 2026 Knee Abnormality Detection AI Challenge
Addresses severe class imbalance across the 12 abnormality findings.
"""

import torch
import torch.nn as nn

class CombinedFocalBCELoss(nn.Module):
    """
    Combined BCE + Focal Loss with focusing parameter gamma and balancing alpha.
    Downweights easy negative examples and forces gradient updates on rare/hard positives.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce_loss
        return loss.mean()
