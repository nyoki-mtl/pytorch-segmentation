from collections import OrderedDict
import torch.nn as nn
from torchvision import models
import pretrainedmodels
# from .inplace_abn import ABN as ActivatedBatchNorm
from .inplace_abn import InPlaceABN as ActivatedBatchNorm


def get_out_channels(layers):
    """access out_channels from last layer of nn.Sequential/list"""
    if hasattr(layers, 'out_channels'):
        return layers.out_channels
    elif isinstance(layers, int):
        return layers
    else:
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            if hasattr(layer, 'out_channels'):
                return layer.out_channels
            elif isinstance(layer, nn.Sequential):
                return get_out_channels(layer)
    raise RuntimeError("cant get_out_channels from {}".format(layers))


def sequential(*args):
    f = nn.Sequential(*args)
    f.out_channels = get_out_channels(args)
    return f


def resnet(name, pretrained=True):
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

    layer0 = sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    layer0[-1].out_channels = resnet.bn1.num_features

    def get_out_channels_from_resnet_block(layer):
        block = layer[-1]
        if isinstance(block, models.resnet.BasicBlock):
            return block.conv2.out_channels
        elif isinstance(block, models.resnet.Bottleneck):
            return block.conv3.out_channels
        raise RuntimeError("unknown resnet block: {}".format(block))

    resnet.layer1.out_channels = resnet.layer1[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer1)
    resnet.layer2.out_channels = resnet.layer2[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer2)
    resnet.layer3.out_channels = resnet.layer3[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer3)
    resnet.layer4.out_channels = resnet.layer4[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer4)
    return [layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]


def resnext(name, pretrained=True):
    if name in ['resnext101_32x4d', 'resnext101_64x4d']:
        pretrained = 'imagenet' if pretrained else None
        resnext = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=pretrained)
    else:
        return NotImplemented

    resnext_features = resnext.features
    layer0 = [resnext_features[i] for i in range(4)]
    layer0 = nn.Sequential(*layer0)
    layer0.out_channels = layer0[-1].out_channels = 64

    layer1 = resnext_features[4]
    layer1.out_channels = layer1[-1].out_channels = 256

    layer2 = resnext_features[5]
    layer2.out_channels = layer2[-1].out_channels = 512

    layer3 = resnext_features[6]
    layer3.out_channels = layer3[-1].out_channels = 1024

    layer4 = resnext_features[7]
    layer4.out_channels = layer4[-1].out_channels = 2048
    return [layer0, layer1, layer2, layer3, layer4]


def replace_bn(bn, act=None):
    slope = 0.01
    if isinstance(act, nn.ReLU):
        activation = 'leaky_relu'  # approximate relu
    elif isinstance(act, nn.LeakyReLU):
        activation = 'leaky_relu'
        slope = act.negative_slope
    elif isinstance(act, nn.ELU):
        activation = 'elu'
    else:
        activation = 'none'
    abn = ActivatedBatchNorm(num_features=bn.num_features,
                             eps=bn.eps,
                             momentum=bn.momentum,
                             affine=bn.affine,
                             activation=activation,
                             slope=slope)
    state_dict = bn.state_dict()
    del state_dict['num_batches_tracked']
    abn.load_state_dict(state_dict)
    return abn


def replace_bn_in_sequential(layer0, block=None):
    layer0_modules = []
    last_bn = None
    for n, m in layer0.named_children():
        if isinstance(m, nn.BatchNorm2d):
            last_bn = (n, m)
        else:
            activation = 'none'
            if last_bn:
                abn = replace_bn(last_bn[1], m)
                activation = abn.activation
                layer0_modules.append((last_bn[0], abn))
                last_bn = None
            if activation == 'none':
                if block and isinstance(m, block):
                    m = replace_bn_in_block(m)
                elif isinstance(m, nn.Sequential):
                    m = replace_bn_in_sequential(m, block)
                layer0_modules.append((n, m))
    if last_bn:
        abn = replace_bn(last_bn[1])
        layer0_modules.append((last_bn[0], abn))
    return nn.Sequential(OrderedDict(layer0_modules))


class DummyModule(nn.Module):
    def forward(self, x):
        return x


def replace_bn_in_block(block):
    block.bn1 = replace_bn(block.bn1, block.relu)
    block.bn2 = replace_bn(block.bn2, block.relu)
    block.bn3 = replace_bn(block.bn3)
    block.relu = DummyModule()
    if block.downsample:
        block.downsample = replace_bn_in_sequential(block.downsample)
    return nn.Sequential(block,
                         nn.ReLU(inplace=True))


def se_net(name, pretrained):
    if name in ['se_resnet50', 'se_resnet101', 'se_resnet152',
                'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154']:
        pretrained = 'imagenet' if pretrained else None
        senet = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=pretrained)
    else:
        return NotImplemented

    layer0 = replace_bn_in_sequential(senet.layer0)

    block = senet.layer1[0].__class__
    layer1 = replace_bn_in_sequential(senet.layer1, block=block)
    layer1.out_channels = layer1[-1].out_channels = senet.layer1[-1].conv3.out_channels
    layer0.out_channels = layer0[-1].out_channels = senet.layer1[0].conv1.in_channels

    layer2 = replace_bn_in_sequential(senet.layer2, block=block)
    layer2.out_channels = layer2[-1].out_channels = senet.layer2[-1].conv3.out_channels

    layer3 = replace_bn_in_sequential(senet.layer3, block=block)
    layer3.out_channels = layer3[-1].out_channels = senet.layer3[-1].conv3.out_channels

    layer4 = replace_bn_in_sequential(senet.layer4, block=block)
    layer4.out_channels = layer4[-1].out_channels = senet.layer4[-1].conv3.out_channels

    return [layer0, layer1, layer2, layer3, layer4]


def create_encoder(enc_type, pretrained):
    if enc_type.startswith('resnet'):
        return resnet(enc_type, pretrained)
    elif enc_type.startswith('resnext'):
        return resnext(enc_type, pretrained)
    elif enc_type.startswith('se'):
        return se_net(enc_type, pretrained)
    else:
        raise NotImplementedError
