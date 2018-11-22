import torch
import torch.nn as nn
import torch.nn.functional as F

from models.net import SegmentationNet

if __name__ == '__main__':
    enc_type = 'resnet50'
    dec_type = 'psp'
    print(enc_type, dec_type)
    model = SegmentationNet(output_channels=21,
                            enc_type=enc_type,
                            dec_type=dec_type,
                            pretrained=False,
                            debug=True)
    x = torch.rand(2, 3, 256, 256)
    print('input', x.size())
    y = model(x)
