import argparse
from pathlib import Path

import tensorflow as tf
import torch
from models.net import SPPNet


def convert_mobilenetv2(ckpt_path, num_classes):
    def conv_converter(pt_layer, tf_layer_name, depthwise=False, bias=False):
        if depthwise:
            pt_layer.weight.data = torch.Tensor(
                reader.get_tensor(f'{tf_layer_name}/depthwise_weights').transpose(2, 3, 0, 1))
        else:
            pt_layer.weight.data = torch.Tensor(reader.get_tensor(f'{tf_layer_name}/weights').transpose(3, 2, 0, 1))

        if bias:
            pt_layer.bias.data = torch.Tensor(reader.get_tensor(f'{tf_layer_name}/biases'))

    def bn_converter(pt_layer, tf_layer_name):
        pt_layer.bias.data = torch.Tensor(reader.get_tensor(f'{tf_layer_name}/beta'))
        pt_layer.weight.data = torch.Tensor(reader.get_tensor(f'{tf_layer_name}/gamma'))
        pt_layer.running_mean.data = torch.Tensor(reader.get_tensor(f'{tf_layer_name}/moving_mean'))
        pt_layer.running_var.data = torch.Tensor(reader.get_tensor(f'{tf_layer_name}/moving_variance'))

    def block_converter(pt_layer, tf_layer_name):
        if hasattr(pt_layer, 'expand'):
            conv_converter(pt_layer.expand.conv, f'{tf_layer_name}/expand')
            bn_converter(pt_layer.expand.bn, f'{tf_layer_name}/expand/BatchNorm')

        conv_converter(pt_layer.depthwise.conv, f'{tf_layer_name}/depthwise', depthwise=True)
        bn_converter(pt_layer.depthwise.bn, f'{tf_layer_name}/depthwise/BatchNorm')

        conv_converter(pt_layer.project.conv, f'{tf_layer_name}/project')
        bn_converter(pt_layer.project.bn, f'{tf_layer_name}/project/BatchNorm')

    reader = tf.train.NewCheckpointReader(ckpt_path)
    model = SPPNet(num_classes, enc_type='mobilenetv2', dec_type='maspp')

    # MobileNetV2
    conv_converter(model.encoder.conv, 'MobilenetV2/Conv')
    bn_converter(model.encoder.bn, 'MobilenetV2/Conv/BatchNorm')

    block_converter(model.encoder.block0, 'MobilenetV2/expanded_conv')
    block_converter(model.encoder.block1, 'MobilenetV2/expanded_conv_1')
    block_converter(model.encoder.block2, 'MobilenetV2/expanded_conv_2')
    block_converter(model.encoder.block3, 'MobilenetV2/expanded_conv_3')
    block_converter(model.encoder.block4, 'MobilenetV2/expanded_conv_4')
    block_converter(model.encoder.block5, 'MobilenetV2/expanded_conv_5')
    block_converter(model.encoder.block6, 'MobilenetV2/expanded_conv_6')
    block_converter(model.encoder.block7, 'MobilenetV2/expanded_conv_7')
    block_converter(model.encoder.block8, 'MobilenetV2/expanded_conv_8')
    block_converter(model.encoder.block9, 'MobilenetV2/expanded_conv_9')
    block_converter(model.encoder.block10, 'MobilenetV2/expanded_conv_10')
    block_converter(model.encoder.block11, 'MobilenetV2/expanded_conv_11')
    block_converter(model.encoder.block12, 'MobilenetV2/expanded_conv_12')
    block_converter(model.encoder.block13, 'MobilenetV2/expanded_conv_13')
    block_converter(model.encoder.block14, 'MobilenetV2/expanded_conv_14')
    block_converter(model.encoder.block15, 'MobilenetV2/expanded_conv_15')
    block_converter(model.encoder.block16, 'MobilenetV2/expanded_conv_16')

    # SPP
    conv_converter(model.spp.aspp0.conv, 'aspp0')
    bn_converter(model.spp.aspp0.bn, 'aspp0/BatchNorm')
    conv_converter(model.spp.image_pooling.conv, 'image_pooling')
    bn_converter(model.spp.image_pooling.bn, 'image_pooling/BatchNorm')
    conv_converter(model.spp.conv, 'concat_projection')
    bn_converter(model.spp.bn, 'concat_projection/BatchNorm')

    # Logits
    conv_converter(model.logits, 'logits/semantic', bias=True)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('num_classes', type=int)
    parser.add_argument('output_path')
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    num_classes = args.num_classes
    output_path = Path(args.output_path)
    output_path.parent.mkdir()

    model = convert_mobilenetv2(ckpt_path, num_classes)
    torch.save(model.state_dict(), output_path)
