import torch
import torch.nn.functional as F
from torch import nn
# from .inplace_abn import ABN as ActivatedBatchNorm
from .inplace_abn import InPlaceABN as ActivatedBatchNorm


class PSP(nn.Module):
    def __init__(self, in_channels, out_channels, pyramids=(1, 2, 3, 6)):
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
    def __init__(self, in_channels, out_channels, pyramids=(1, 6, 12, 18)):
        super().__init__()
        stages = []
        for p in pyramids:
            if p == 1:
                kernel_size = 1
                padding = 0
                dilation = p
            else:
                kernel_size = 3
                padding = p
                dilation = p
            stages.append(nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                  kernel_size, 1, padding, dilation, bias=False),
                                        ActivatedBatchNorm(out_channels)))
        self.stages = nn.ModuleList(stages)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                  ActivatedBatchNorm(out_channels))

    def forward(self, x):
        x_size = x.size()
        out = self.pool(x)
        out = [F.interpolate(out, size=x_size[2:], mode="bilinear", align_corners=False)]
        for stage in self.stages:
            out.append(stage(x))
        out = torch.cat(out, dim=1)
        return out
