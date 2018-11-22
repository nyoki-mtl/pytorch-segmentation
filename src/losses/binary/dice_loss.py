import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, preds, labels):
        return 1 - (2 * torch.sum(preds * labels) + self.smooth) / \
                        (torch.sum(preds) + torch.sum(labels) + self.smooth + self.eps)


class MixedDiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.2, bce_weight=0.9):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)
        loss = self.dice_weight * self.dice_loss(preds, labels) + self.bce_weight * self.bce_loss(preds, labels)
        return loss
