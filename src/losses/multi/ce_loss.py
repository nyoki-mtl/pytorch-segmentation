import torch
import torch.nn as nn


class CrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=255, weight=None, size_average=True, batch_average=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)

    def forward(self, preds, labels):
        n, c, h, w = preds.size()
        loss = self.loss_fn(preds, labels)
        if self.size_average:
            loss /= h * w
        if self.batch_average:
            loss /= n
        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, weight=None, thresh=0.7):
        super().__init__()
        self.ignore_label = ignore_label
        self.weight = weight
        self.thresh = thresh
        self.loss_fn = nn.NLLLoss()
        self.softmax = nn.LogSoftmax()

    def forward(self, preds, labels):
        # n, c, h, w = preds.size()
        preds = self.softmax(preds)

        loss = torch.Tensor(1).zero_()
        for idx, pred in enumerate(preds):
            pred = pred.unsqueeze(0)
            label = labels[idx]
            loss = torch.cat([loss, self.loss_fn(pred, label)], 0)

        loss = loss[1:]
        # if self.thresh == 1:
        #     valid_loss = loss

        index = torch.topk(loss, int(self.thresh * loss.size()[0]))
        valid_loss = loss[index[1]]

        return torch.mean(valid_loss)
