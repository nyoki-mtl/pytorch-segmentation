import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


class ExpandedConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 expand_ratio=6, skip_connection=False):
        super().__init__()

        self.stride = stride
        self.kernel_size = 3
        self.dilation = dilation
        self.expand_ratio = expand_ratio
        self.skip_connection = skip_connection
        middle_channels = in_channels * expand_ratio

        if self.expand_ratio != 1:
            # pointwise
            self.expand = nn.Sequential(OrderedDict(
                [('conv', nn.Conv2d(in_channels, middle_channels, 1, bias=False)),
                 ('bn', nn.BatchNorm2d(middle_channels)),
                 ('relu', nn.ReLU6(inplace=True))
                 ]))

        # depthwise
        self.depthwise = nn.Sequential(OrderedDict(
            [('conv', nn.Conv2d(middle_channels, middle_channels, 3, stride, dilation, dilation, groups=middle_channels, bias=False)),
             ('bn', nn.BatchNorm2d(middle_channels)),
             ('relu', nn.ReLU6(inplace=True))
             ]))

        # project
        self.project = nn.Sequential(OrderedDict(
            [('conv', nn.Conv2d(middle_channels, out_channels, 1, bias=False)),
             ('bn', nn.BatchNorm2d(out_channels))
             ]))

    def forward(self, x):
        if self.expand_ratio != 1:
            residual = self.project(self.depthwise(self.expand(x)))
        else:
            residual = self.project(self.depthwise(x))

        if self.skip_connection:
            outputs = x + residual
        else:
            outputs = residual
        return outputs


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=False, model_path='../model/mobilenetv2_encoder/model.pth'):
        super().__init__()

        self.conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6()

        self.block0 = ExpandedConv(32, 16, expand_ratio=1)
        self.block1 = ExpandedConv(16, 24, stride=2)
        self.block2 = ExpandedConv(24, 24, skip_connection=True)
        self.block3 = ExpandedConv(24, 32, stride=2)
        self.block4 = ExpandedConv(32, 32, skip_connection=True)
        self.block5 = ExpandedConv(32, 32, skip_connection=True)
        self.block6 = ExpandedConv(32, 64)
        self.block7 = ExpandedConv(64, 64, dilation=2, skip_connection=True)
        self.block8 = ExpandedConv(64, 64, dilation=2, skip_connection=True)
        self.block9 = ExpandedConv(64, 64, dilation=2, skip_connection=True)
        self.block10 = ExpandedConv(64, 96, dilation=2)
        self.block11 = ExpandedConv(96, 96, dilation=2, skip_connection=True)
        self.block12 = ExpandedConv(96, 96, dilation=2, skip_connection=True)
        self.block13 = ExpandedConv(96, 160, dilation=2)
        self.block14 = ExpandedConv(160, 160, dilation=4, skip_connection=True)
        self.block15 = ExpandedConv(160, 160, dilation=4, skip_connection=True)
        self.block16 = ExpandedConv(160, 320, dilation=4)

        if pretrained:
            self.load_pretrained_model(model_path)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        return x

    def load_pretrained_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print(f'Load from {model_path}!')
