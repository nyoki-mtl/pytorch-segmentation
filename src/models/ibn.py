import torch
import torch.nn as nn
from .common import ActivatedBatchNorm


class IBN(nn.Module):
    def __init__(self, planes):
        super().__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.Sequential(nn.InstanceNorm2d(half1, affine=True),
                                nn.ReLU(inplace=True))
        self.BN = ActivatedBatchNorm(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class ImprovedIBNaDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            IBN(in_channels // 4),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1),
            ActivatedBatchNorm(in_channels // 4),
            nn.Conv2d(in_channels // 4, out_channels, 1),
            ActivatedBatchNorm(out_channels)
        )

    def forward(self, x):
        return self.block(x)
