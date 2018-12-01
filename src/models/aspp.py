from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import SeparableConv2d


class ASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(2048, 256, 1, bias=False)),
                                                ('bn', nn.BatchNorm2d(256)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.aspp1 = SeparableConv2d(2048, 256, dilation=12, relu_first=False)
        self.aspp2 = SeparableConv2d(2048, 256, dilation=24, relu_first=False)
        self.aspp3 = SeparableConv2d(2048, 256, dilation=36, relu_first=False)

        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))),
                                                        ('conv', nn.Conv2d(2048, 256, 1, bias=False)),
                                                        ('bn', nn.BatchNorm2d(256)),
                                                        ('relu', nn.ReLU(inplace=True))]))

        self.conv = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)

        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = torch.cat((pool, x0, x1, x2, x3), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x
