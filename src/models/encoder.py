import torch.nn as nn
from torchvision import models
import pretrainedmodels
from .xception import Xception65
from .mobilenet import MobileNetV2


def resnet(name, pretrained=False):
    def get_channels(layer):
        block = layer[-1]
        if isinstance(block, models.resnet.BasicBlock):
            return block.conv2.out_channels
        elif isinstance(block, models.resnet.Bottleneck):
            return block.conv3.out_channels
        raise RuntimeError("unknown resnet block: {}".format(block))

    if name == 'resnet18':
        resnet = models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        resnet = models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        resnet = models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        resnet = models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        resnet = models.resnet152(pretrained=pretrained)
    else:
        return NotImplemented

    layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    layer0.out_channels = resnet.bn1.num_features
    resnet.layer1.out_channels = get_channels(resnet.layer1)
    resnet.layer2.out_channels = get_channels(resnet.layer2)
    resnet.layer3.out_channels = get_channels(resnet.layer3)
    resnet.layer4.out_channels = get_channels(resnet.layer4)
    return [layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]


def resnext(name, pretrained=False):
    if name in ['resnext101_32x4d', 'resnext101_64x4d']:
        pretrained = 'imagenet' if pretrained else None
        resnext = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=pretrained)
    else:
        return NotImplemented

    layer0 = nn.Sequential(resnext.features[0],
                           resnext.features[1],
                           resnext.features[2],
                           resnext.features[3])
    layer1 = resnext.features[4]
    layer2 = resnext.features[5]
    layer3 = resnext.features[6]
    layer4 = resnext.features[7]

    layer0.out_channels = 64
    layer1.out_channels = 256
    layer2.out_channels = 512
    layer3.out_channels = 1024
    layer4.out_channels = 2048
    return [layer0, layer1, layer2, layer3, layer4]


def se_net(name, pretrained=False):
    if name in ['se_resnet50', 'se_resnet101', 'se_resnet152',
                'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154']:
        pretrained = 'imagenet' if pretrained else None
        senet = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=pretrained)
    else:
        return NotImplemented

    layer0 = senet.layer0
    layer1 = senet.layer1
    layer2 = senet.layer2
    layer3 = senet.layer3
    layer4 = senet.layer4

    layer0.out_channels = senet.layer1[0].conv1.in_channels
    layer1.out_channels = senet.layer1[-1].conv3.out_channels
    layer2.out_channels = senet.layer2[-1].conv3.out_channels
    layer3.out_channels = senet.layer3[-1].conv3.out_channels
    layer4.out_channels = senet.layer4[-1].conv3.out_channels

    return [layer0, layer1, layer2, layer3, layer4]


def create_encoder(enc_type, output_stride=8, pretrained=True):
    if enc_type.startswith('resnet'):
        return resnet(enc_type, pretrained)
    elif enc_type.startswith('resnext'):
        return resnext(enc_type, pretrained)
    elif enc_type.startswith('se'):
        return se_net(enc_type, pretrained)
    elif enc_type == 'xception65':
        return Xception65(output_stride)
    elif enc_type == 'mobilenetv2':
        return MobileNetV2(pretrained)
    else:
        raise NotImplementedError
