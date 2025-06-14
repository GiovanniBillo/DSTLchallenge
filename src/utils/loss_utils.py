import torch
import torch.nn as nn

class JaccardLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        # Apply sigmoid to convert logits to probabilities
        preds = torch.sigmoid(preds)

        # Flatten to (B, C, H*W)
        preds = preds.view(preds.size(0), preds.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (preds * targets).sum(dim=2)
        union = preds.sum(dim=2) + targets.sum(dim=2) - intersection

        jaccard = (intersection + self.eps) / (union + self.eps)
        return 1 - jaccard.mean()


class BCEJaccardLoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.jaccard = JaccardLoss(eps)
        self.alpha = alpha

    def forward(self, preds, targets):
        return self.alpha * self.bce(preds, targets) + (1 - self.alpha) * self.jaccard(preds, targets)

