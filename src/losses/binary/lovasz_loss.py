"""
https://arxiv.org/abs/1705.08790
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def hinge(pred, label):
    signs = 2 * label - 1
    errors = 1 - pred * signs
    return errors


def lovasz_loss(preds, labels):
    preds = preds.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    errors = hinge(preds, labels)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


class LovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = lovasz_loss

    def forward(self, preds, labels):
        return self.loss_fn(preds, labels)
