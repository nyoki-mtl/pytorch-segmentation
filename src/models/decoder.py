import torch
import torch.nn as nn
from .ibn import ImprovedIBNaDecoderBlock
from .scse import SELayer, SCSEBlock
from .oc import BaseOC, ASP_OC
from .psp import PSP, ASPP
# from .inplace_abn import ABN as ActivatedBatchNorm
from .inplace_abn import InPlaceABN as ActivatedBatchNorm


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
            BaseOC(in_channels=middle_channels, out_channels=middle_channels,
                   key_channels=middle_channels // 2,
                   value_channels=middle_channels // 2,
                   dropout=0.2),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class DecoderOCBase(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ActivatedBatchNorm(out_channels),
            BaseOC(in_channels=out_channels, out_channels=out_channels,
                   key_channels=out_channels // 2, value_channels=out_channels // 2))

    def forward(self, x):
        return self.block(x)


def create_decoder(dec_type):
    if dec_type == 'unet_scse':
        return DecoderUnetSCSE
    elif dec_type == 'unet_seibn':
        return DecoderUnetSEIBN
    elif dec_type == 'unet_oc':
        return DecoderUnetOC
    elif dec_type == 'oc_base':
        return DecoderOCBase
    elif dec_type == 'oc_asp':
        return ASP_OC
    elif dec_type == 'psp':
        return PSP
    elif dec_type == 'aspp':
        return ASPP
    else:
        raise NotImplementedError
