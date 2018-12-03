import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import ActivatedBatchNorm, SeparableConv2d
from .ibn import ImprovedIBNaDecoderBlock
from .scse import SELayer, SCSEBlock
from .oc import BaseOC


class DecoderUnetSCSE(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            ActivatedBatchNorm(middle_channels),
            SCSEBlock(middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class DecoderUnetSEIBN(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            SELayer(in_channels),
            ImprovedIBNaDecoderBlock(in_channels, out_channels)
        )

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class DecoderUnetOC(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            ActivatedBatchNorm(middle_channels),
            BaseOC(in_channels=middle_channels,
                   out_channels=middle_channels,
                   dropout=0.2),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class DecoderSPP(nn.Module):
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


def create_decoder(dec_type):
    if dec_type == 'unet_scse':
        return DecoderUnetSCSE
    elif dec_type == 'unet_seibn':
        return DecoderUnetSEIBN
    elif dec_type == 'unet_oc':
        return DecoderUnetOC
    else:
        raise NotImplementedError
