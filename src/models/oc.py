"""
OCNet: Object Context Network for Scene Parsing
https://github.com/PkuRainBow/OCNet
"""

import torch
from torch import nn
from torch.nn import functional as F
from .common import ActivatedBatchNorm


class SelfAttentionBlock2D(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1),
            ActivatedBatchNorm(self.key_channels)
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)  # b, h*w, v
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)  # b, h*w, k
        key = self.f_key(x).view(batch_size, self.key_channels, -1)  # b, k, h*w

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # b * h*w * h*w

        context = torch.matmul(sim_map, value)  # b * h*w * v
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class BaseOC_Context(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout=0.05, sizes=(1,)):
        super().__init__()
        self.stages = nn.ModuleList(
            [SelfAttentionBlock2D(in_channels, key_channels, value_channels, out_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            ActivatedBatchNorm(out_channels),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


class BaseOC(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ActivatedBatchNorm(out_channels),
            BaseOC_Context(in_channels=out_channels, out_channels=out_channels,
                           key_channels=out_channels // 2, value_channels=out_channels // 2, dropout=dropout))

    def forward(self, x):
        return self.block(x)


class ASPOC(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, output_stride=8):
        super().__init__()
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        else:
            raise NotImplementedError

        self.context = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=True),
            ActivatedBatchNorm(out_channels),
            BaseOC_Context(in_channels=out_channels, out_channels=out_channels,
                           key_channels=out_channels // 2, value_channels=out_channels,
                           dropout=0, sizes=([2])))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=False),
            ActivatedBatchNorm(out_channels))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[0], dilation=dilations[0],
                      bias=False),
            ActivatedBatchNorm(out_channels))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[1], dilation=dilations[1],
                      bias=False),
            ActivatedBatchNorm(out_channels))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[2], dilation=dilations[2],
                      bias=False),
            ActivatedBatchNorm(out_channels))
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, padding=0, dilation=1, bias=False),
            ActivatedBatchNorm(out_channels),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        output = self.conv_bn_dropout(out)
        return output
