import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import create_encoder
from .decoder import create_decoder
from .tta import SegmentatorTTA
# from .inplace_abn import ABN as ActivatedBatchNorm
from .inplace_abn import InPlaceABN as ActivatedBatchNorm


class SegmentationNet(nn.Module, SegmentatorTTA):
    def __init__(self, output_channels=21, enc_type='resnet50', dec_type='oc_base',
                 num_filters=16, pretrained=True):
        super().__init__()
        self.output_channels = output_channels
        self.enc_type = enc_type
        self.dec_type = dec_type

        if dec_type.startswith('unet'):
            assert enc_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                'resnext101_32x4d', 'resnext101_64x4d',
                                'se_resnet50', 'se_resnet101', 'se_resnet152',
                                'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154']
            assert dec_type in ['unet_scse', 'unet_seibn', 'unet_oc']
        elif dec_type.startswith(('oc', 'psp', 'aspp')):
            assert enc_type in ['resnet50', 'resnet101', 'resnet152', 'senet154']
            assert dec_type in ['oc_base', 'oc_asp', 'psp', 'aspp']
        else:
            raise NotImplementedError

        encoder = create_encoder(enc_type, pretrained)
        Decoder = create_decoder(dec_type)

        self.encoder1 = encoder[0]
        self.encoder2 = encoder[1]
        self.encoder3 = encoder[2]
        self.encoder4 = encoder[3]
        self.encoder5 = encoder[4]

        if dec_type.startswith('unet'):
            self.pool = nn.MaxPool2d(2, 2)
            self.center = Decoder(self.encoder5.out_channels, num_filters * 32 * 2, num_filters * 32)

            self.decoder5 = Decoder(self.encoder5.out_channels + num_filters * 32, num_filters * 32 * 2,
                                    num_filters * 16)
            self.decoder4 = Decoder(self.encoder4.out_channels + num_filters * 16, num_filters * 16 * 2,
                                    num_filters * 8)
            self.decoder3 = Decoder(self.encoder3.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 4)
            self.decoder2 = Decoder(self.encoder2.out_channels + num_filters * 4, num_filters * 4 * 2, num_filters * 2)
            self.decoder1 = Decoder(self.encoder1.out_channels + num_filters * 2, num_filters * 2 * 2, num_filters)

            self.output = nn.Sequential(
                nn.Conv2d(num_filters * (16 + 8 + 4 + 2 + 1), 64, kernel_size=1, padding=0),
                ActivatedBatchNorm(64),
                nn.Conv2d(64, self.output_channels, kernel_size=1)
            )

        elif dec_type.startswith(('oc', 'psp', 'aspp')):
            # output: n, class, h//8, w//8
            for n, m in self.encoder4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.encoder5.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

            if dec_type.startswith(('oc', 'psp')):
                self.decoder = Decoder(2048, 512)
                self.output = nn.Sequential(
                    nn.Conv2d(512, 64, kernel_size=1, padding=0),
                    ActivatedBatchNorm(64),
                    nn.Conv2d(64, self.output_channels, kernel_size=1)
                )
            elif dec_type.startswith('aspp'):
                self.decoder = Decoder(2048, 256)
                self.output = nn.Sequential(
                    nn.Conv2d(1280, 256, kernel_size=1, bias=False),
                    ActivatedBatchNorm(256),
                    nn.Conv2d(256, self.output_channels, kernel_size=1)
                )

    def forward(self, x):
        img_size = x.shape[2:]

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        if self.dec_type.startswith('unet'):
            c = self.center(self.pool(e5))
            e1_up = F.interpolate(e1, scale_factor=2, mode='bilinear', align_corners=False)

            d5 = self.decoder5(c, e5)
            d4 = self.decoder4(d5, e4)
            d3 = self.decoder3(d4, e3)
            d2 = self.decoder2(d3, e2)
            d1 = self.decoder1(d2, e1_up)

            u5 = F.interpolate(d5, img_size, mode='bilinear', align_corners=False)
            u4 = F.interpolate(d4, img_size, mode='bilinear', align_corners=False)
            u3 = F.interpolate(d3, img_size, mode='bilinear', align_corners=False)
            u2 = F.interpolate(d2, img_size, mode='bilinear', align_corners=False)

            # Hyper column
            d = torch.cat((d1, u2, u3, u4, u5), 1)
            output = self.output(d)

        elif self.dec_type.startswith(('oc', 'psp', 'aspp')):
            d = self.decoder(e5)
            output = self.output(d)
            output = F.interpolate(output, img_size, mode='bilinear', align_corners=False)

        return output
