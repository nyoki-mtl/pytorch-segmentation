import torch
import torch.nn as nn
import torch.nn.functional as F
from .xception import Xception65
from .aspp import ASPP
from .common import SeparableConv2d


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 48, 1, bias=False)
        self.bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.sep1 = SeparableConv2d(304, 256, relu_first=False)
        self.sep2 = SeparableConv2d(256, 256, relu_first=False)

    def forward(self, x, low_level_feat):
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        low_level_feat = self.conv(low_level_feat)
        low_level_feat = self.bn(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.sep1(x)
        x = self.sep2(x)
        return x


class DeepLab(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.backbone = Xception65()
        self.aspp = ASPP()
        self.decoder = Decoder()
        self.logits = nn.Conv2d(256, num_classes, 1)

    def forward(self, inputs):
        x, low_level_feat = self.backbone(inputs)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = self.logits(x)
        return x

    def update_bn_eps(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eps = 1e-3

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder, self.logits]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
