import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, valid_mask=None):
        if valid_mask is not None:
            loss = 0
            batch_size = logits.shape[0]
            for logit, lbl, val_msk in zip(logits, labels, valid_mask):
                logit = logit[:, val_msk]
                lbl = lbl[:, val_msk]
                loss -= torch.mean(torch.mul(F.log_softmax(logit, dim=0), F.softmax(lbl, dim=0)))
            return loss / batch_size
        else:
            return torch.mean(torch.mul(F.log_softmax(logits, dim=1), F.softmax(labels, dim=1)))


class KlLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, valid_mask=None):
        if valid_mask is not None:
            loss = 0
            batch_size = logits.shape[0]
            for logit, lbl, val_msk in zip(logits, labels, valid_mask):
                logit = logit[:, val_msk]
                lbl = lbl[:, val_msk]
                loss += torch.mean(F.kl_div(F.log_softmax(logit, dim=0), F.softmax(lbl, dim=0), reduction='none'))
            return loss / batch_size
        else:
            return torch.mean(F.kl_div(F.log_softmax(logits, dim=1), F.softmax(labels, dim=1), reduction='none'))
