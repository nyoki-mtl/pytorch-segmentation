from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from .common import ActivatedBatchNorm, SeparableConv2d
from .oc import BaseOC, ASPOC


class SPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, pyramids=(1, 2, 3, 6)):
        super().__init__()
        stages = []
        for p in pyramids:
            stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(p),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                ActivatedBatchNorm(out_channels)
            ))
        self.stages = nn.ModuleList(stages)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pyramids), out_channels, kernel_size=1),
            ActivatedBatchNorm(out_channels)
        )

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for stage in self.stages:
            out.append(F.interpolate(stage(x), size=x_size[2:], mode='bilinear', align_corners=False))
        out = self.bottleneck(torch.cat(out, 1))
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, output_stride=8):
        super().__init__()
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        else:
            raise NotImplementedError
        # dilations = [6, 12, 18]

        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                ('bn', nn.BatchNorm2d(out_channels)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.aspp1 = SeparableConv2d(in_channels, out_channels, dilation=dilations[0], relu_first=False)
        self.aspp2 = SeparableConv2d(in_channels, out_channels, dilation=dilations[1], relu_first=False)
        self.aspp3 = SeparableConv2d(in_channels, out_channels, dilation=dilations[2], relu_first=False)

        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))),
                                                        ('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                        ('bn', nn.BatchNorm2d(out_channels)),
                                                        ('relu', nn.ReLU(inplace=True))]))

        self.conv = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
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


class MobileASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(320, 256, 1, bias=False)),
                                                ('bn', nn.BatchNorm2d(256)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))),
                                                        ('conv', nn.Conv2d(320, 256, 1, bias=False)),
                                                        ('bn', nn.BatchNorm2d(256)),
                                                        ('relu', nn.ReLU(inplace=True))]))

        self.conv = nn.Conv2d(512, 256, 1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = self.aspp0(x)
        x = torch.cat((pool, x), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class SPPDecoder(nn.Module):
    def __init__(self, in_channels, reduced_layer_num=48):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, reduced_layer_num, 1, bias=False)
        self.bn = nn.BatchNorm2d(reduced_layer_num)
        self.relu = nn.ReLU(inplace=True)
        self.sep1 = SeparableConv2d(256+reduced_layer_num, 256, relu_first=False)
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


def create_spp(dec_type, in_channels=2048, middle_channels=256, output_stride=8):
    if dec_type == 'spp':
        return SPP(in_channels, middle_channels), SPPDecoder(middle_channels)
    elif dec_type == 'aspp':
        return ASPP(in_channels, middle_channels, output_stride), SPPDecoder(middle_channels)
    elif dec_type == 'oc_base':
        return BaseOC(in_channels, middle_channels), SPPDecoder(middle_channels)
    elif dec_type in 'oc_asp':
        return ASPOC(in_channels, middle_channels, output_stride), SPPDecoder(middle_channels)
    else:
        raise NotImplementedError


def create_mspp(dec_type):
    if dec_type == 'spp':
        return SPP(320, 256)
    elif dec_type == 'aspp':
        return ASPP(320, 256, 8)
    elif dec_type == 'oc_base':
        return BaseOC(320, 256)
    elif dec_type == 'oc_asp':
        return ASPOC(320, 256, 8)
    elif dec_type == 'maspp':
        return MobileASPP()
    elif dec_type == 'maspp_dec':
        return MobileASPP(), SPPDecoder(24, reduced_layer_num=12)
    else:
        raise NotImplementedError
