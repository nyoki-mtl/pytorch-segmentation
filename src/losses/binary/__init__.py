import torch.nn as nn

from .focal_loss import FocalLoss
from .lovasz_loss import LovaszLoss
from .dice_loss import DiceLoss, MixedDiceBCELoss


class BinaryClassCriterion(nn.Module):
    def __init__(self, loss_type='BCE', **kwargs):
        super().__init__()
        if loss_type == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss(**kwargs)
        elif loss_type == 'Focal':
            self.criterion = FocalLoss(**kwargs)
        elif loss_type == 'Lovasz':
            self.criterion = LovaszLoss(**kwargs)
        elif loss_type == 'Dice':
            self.criterion = DiceLoss(**kwargs)
        elif loss_type == 'MixedDiceBCE':
            self.criterion = MixedDiceBCELoss(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss
