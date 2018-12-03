import torch
import torch.nn as nn


class CrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=255, weight=None, size_average=True, batch_average=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')

    def forward(self, preds, labels):
        n, c, h, w = preds.size()
        loss = self.loss_fn(preds, labels)
        if self.size_average:
            loss /= h * w
        if self.batch_average:
            loss /= n
        return loss
