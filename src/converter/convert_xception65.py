import argparse
from pathlib import Path

import tensorflow as tf
import torch
from models.net import SPPNet


def convert_xception65(ckpt_path, num_classes):
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

    def sepconv_converter(pt_layer, tf_layer_name):
        conv_converter(pt_layer.depthwise, f'{tf_layer_name}_depthwise', True)
        bn_converter(pt_layer.bn_depth, f'{tf_layer_name}_depthwise/BatchNorm')
        conv_converter(pt_layer.pointwise, f'{tf_layer_name}_pointwise')
        bn_converter(pt_layer.bn_point, f'{tf_layer_name}_pointwise/BatchNorm')

    def block_converter(pt_block, tf_block_name):
        if pt_block.skip_connection_type == 'conv':
            conv_converter(pt_block.conv, f'{tf_block_name}/shortcut')
            bn_converter(pt_block.bn, f'{tf_block_name}/shortcut/BatchNorm')

        sepconv_converter(pt_block.sep_conv1.block, f'{tf_block_name}/separable_conv1')
        sepconv_converter(pt_block.sep_conv2.block, f'{tf_block_name}/separable_conv2')
        sepconv_converter(pt_block.sep_conv3.block, f'{tf_block_name}/separable_conv3')

    reader = tf.train.NewCheckpointReader(ckpt_path)
    model = SPPNet(num_classes, enc_type='xception65', dec_type='aspp', output_stride=8)

    # Xception
    ## Entry flow
    conv_converter(model.encoder.conv1, 'xception_65/entry_flow/conv1_1')
    bn_converter(model.encoder.bn1, 'xception_65/entry_flow/conv1_1/BatchNorm')
    conv_converter(model.encoder.conv2, 'xception_65/entry_flow/conv1_2')
    bn_converter(model.encoder.bn2, 'xception_65/entry_flow/conv1_2/BatchNorm')
    block_converter(model.encoder.block1, 'xception_65/entry_flow/block1/unit_1/xception_module')
    block_converter(model.encoder.block2, 'xception_65/entry_flow/block2/unit_1/xception_module')
    block_converter(model.encoder.block3, 'xception_65/entry_flow/block3/unit_1/xception_module')
    ## Middle flow
    block_converter(model.encoder.block4, 'xception_65/middle_flow/block1/unit_1/xception_module')
    block_converter(model.encoder.block5, 'xception_65/middle_flow/block1/unit_2/xception_module')
    block_converter(model.encoder.block6, 'xception_65/middle_flow/block1/unit_3/xception_module')
    block_converter(model.encoder.block7, 'xception_65/middle_flow/block1/unit_4/xception_module')
    block_converter(model.encoder.block8, 'xception_65/middle_flow/block1/unit_5/xception_module')
    block_converter(model.encoder.block9, 'xception_65/middle_flow/block1/unit_6/xception_module')
    block_converter(model.encoder.block10, 'xception_65/middle_flow/block1/unit_7/xception_module')
    block_converter(model.encoder.block11, 'xception_65/middle_flow/block1/unit_8/xception_module')
    block_converter(model.encoder.block12, 'xception_65/middle_flow/block1/unit_9/xception_module')
    block_converter(model.encoder.block13, 'xception_65/middle_flow/block1/unit_10/xception_module')
    block_converter(model.encoder.block14, 'xception_65/middle_flow/block1/unit_11/xception_module')
    block_converter(model.encoder.block15, 'xception_65/middle_flow/block1/unit_12/xception_module')
    block_converter(model.encoder.block16, 'xception_65/middle_flow/block1/unit_13/xception_module')
    block_converter(model.encoder.block17, 'xception_65/middle_flow/block1/unit_14/xception_module')
    block_converter(model.encoder.block18, 'xception_65/middle_flow/block1/unit_15/xception_module')
    block_converter(model.encoder.block19, 'xception_65/middle_flow/block1/unit_16/xception_module')
    ## Exit flow
    block_converter(model.encoder.block20, 'xception_65/exit_flow/block1/unit_1/xception_module')
    block_converter(model.encoder.block21, 'xception_65/exit_flow/block2/unit_1/xception_module')

    # ASPP
    conv_converter(model.spp.aspp0.conv, 'aspp0')
    bn_converter(model.spp.aspp0.bn, 'aspp0/BatchNorm')
    sepconv_converter(model.spp.aspp1.block, 'aspp1')
    sepconv_converter(model.spp.aspp2.block, 'aspp2')
    sepconv_converter(model.spp.aspp3.block, 'aspp3')

    conv_converter(model.spp.image_pooling.conv, 'image_pooling')
    bn_converter(model.spp.image_pooling.bn, 'image_pooling/BatchNorm')
    conv_converter(model.spp.conv, 'concat_projection')
    bn_converter(model.spp.bn, 'concat_projection/BatchNorm')

    # Decoder
    conv_converter(model.decoder.conv, 'decoder/feature_projection0')
    bn_converter(model.decoder.bn, 'decoder/feature_projection0/BatchNorm')

    sepconv_converter(model.decoder.sep1.block, 'decoder/decoder_conv0')
    sepconv_converter(model.decoder.sep2.block, 'decoder/decoder_conv1')

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

    model = convert_xception65(ckpt_path, num_classes)
    torch.save(model.state_dict(), output_path)
