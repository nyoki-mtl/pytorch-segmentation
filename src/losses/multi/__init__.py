import torch.nn as nn

from .focal_loss import FocalLoss
from .lovasz_loss import LovaszSoftmax
from .ohem_loss import OhemCrossEntropy2d
from .softiou_loss import SoftIoULoss


class MultiClassCriterion(nn.Module):
    def __init__(self, loss_type='CrossEntropy', **kwargs):
        super().__init__()
        if loss_type == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'Focal':
            self.criterion = FocalLoss(**kwargs)
        elif loss_type == 'Lovasz':
            self.criterion = LovaszSoftmax(**kwargs)
        elif loss_type == 'OhemCrossEntropy':
            self.criterion = OhemCrossEntropy2d(**kwargs)
        elif loss_type == 'SoftIOU':
            self.criterion = SoftIoULoss(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss
