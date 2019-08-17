import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ActivatedBatchNorm
from .encoder import create_encoder
from .decoder import create_decoder
from .spp import create_spp, create_mspp
from .tta import SegmentatorTTA


class EncoderDecoderNet(nn.Module, SegmentatorTTA):
    def __init__(self, output_channels=19, enc_type='resnet50', dec_type='unet_scse',
                 num_filters=16, pretrained=False):
        super().__init__()
        self.output_channels = output_channels
        self.enc_type = enc_type
        self.dec_type = dec_type

        assert enc_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                            'resnext101_32x4d', 'resnext101_64x4d',
                            'se_resnet50', 'se_resnet101', 'se_resnet152',
                            'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154']
        assert dec_type in ['unet_scse', 'unet_seibn', 'unet_oc']

        encoder = create_encoder(enc_type, pretrained)
        Decoder = create_decoder(dec_type)

        self.encoder1 = encoder[0]
        self.encoder2 = encoder[1]
        self.encoder3 = encoder[2]
        self.encoder4 = encoder[3]
        self.encoder5 = encoder[4]

        self.pool = nn.MaxPool2d(2, 2)
        self.center = Decoder(self.encoder5.out_channels, num_filters * 32 * 2, num_filters * 32)

        self.decoder5 = Decoder(self.encoder5.out_channels + num_filters * 32, num_filters * 32 * 2,
                                num_filters * 16)
        self.decoder4 = Decoder(self.encoder4.out_channels + num_filters * 16, num_filters * 16 * 2,
                                num_filters * 8)
        self.decoder3 = Decoder(self.encoder3.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 4)
        self.decoder2 = Decoder(self.encoder2.out_channels + num_filters * 4, num_filters * 4 * 2, num_filters * 2)
        self.decoder1 = Decoder(self.encoder1.out_channels + num_filters * 2, num_filters * 2 * 2, num_filters)

        self.logits = nn.Sequential(
            nn.Conv2d(num_filters * (16 + 8 + 4 + 2 + 1), 64, kernel_size=1, padding=0),
            ActivatedBatchNorm(64),
            nn.Conv2d(64, self.output_channels, kernel_size=1)
        )

    def forward(self, x):
        img_size = x.shape[2:]

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

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
        logits = self.logits(d)

        return logits


class SPPNet(nn.Module, SegmentatorTTA):
    def __init__(self, output_channels=19, enc_type='xception65', dec_type='aspp', output_stride=8):
        super().__init__()
        self.output_channels = output_channels
        self.enc_type = enc_type
        self.dec_type = dec_type

        assert enc_type in ['xception65', 'mobilenetv2']
        assert dec_type in ['oc_base', 'oc_asp', 'spp', 'aspp', 'maspp']

        self.encoder = create_encoder(enc_type, output_stride=output_stride, pretrained=False)
        if enc_type == 'mobilenetv2':
            self.spp = create_mspp(dec_type)
        else:
            self.spp, self.decoder = create_spp(dec_type, output_stride=output_stride)
        self.logits = nn.Conv2d(256, output_channels, 1)

    def forward(self, inputs):
        if self.enc_type == 'mobilenetv2':
            x = self.encoder(inputs)
            x = self.spp(x)
            x = self.logits(x)
            return x
        else:
            x, low_level_feat = self.encoder(inputs)
            x = self.spp(x)
            x = self.decoder(x, low_level_feat)
            x = self.logits(x)
            return x

    def update_bn_eps(self):
        for m in self.encoder.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eps = 1e-3

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                # for p in m.parameters():
                #     p.requires_grad = False

    def get_1x_lr_params(self):
        for p in self.encoder.parameters():
            yield p

    def get_10x_lr_params(self):
        modules = [self.spp, self.logits]
        if hasattr(self, 'decoder'):
            modules.append(self.decoder)

        for module in modules:
            for p in module.parameters():
                yield p
