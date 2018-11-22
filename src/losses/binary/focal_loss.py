"""
https://arxiv.org/abs/1708.02002
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True, ignore=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.ignore = ignore

    def forward(self, preds, labels):
        mask = labels != self.ignore
        labels = labels[mask]
        preds = preds[mask]
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(preds, labels, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss
