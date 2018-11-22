# PytorchSegmentation
This repository implements general network for semantic segmentation.
You can run various networks like UNet, PSPNet, ASPP, etc., just by writing the config file.

```
python train.py ../config/test.yaml
```

## Directory tree
```
.
├── config
├── data
│   ├── cityscapes
│   │   ├── gtFine
│   │   └── leftImg8bit
│   └── pascal_voc_2012
│        └── VOCdevkit
│            └── VOC2012
│                ├── JPEGImages
│                ├── SegmentationClass
│                └── SegmentationClassAug
├── logs
├── model
└── src
    ├── dataset
    ├── logger
    ├── losses
    │   ├── binary
    │   └── multi
    ├── models
    │   └── inplace_abn
    ├── start_train.sh
    ├── stop_train.sh
    ├── train.py
    └── utils
```

## Networks
### UNet
- encoder type
    - resnet18
    - resnet34
    - resnet50
    - resnet101
    - resnet152
    - resnext101_32x4d
    - resnext101_64x4d
    - se_resnet50
    - se_resnet101
    - se_resnet152
    - se_resnext50_32x4d
    - se_resnext101_32x4d
    - senet154
- decoder type
    - unet_scse
    - unet_seibn
    - unet_oc

### PSPNet
- encoder type
    - resnet18
    - resnet34
    - resnet50
    - resnet101
    - resnet152
    - senet154
- decoder type
    - psp

### ASPP
- encoder type
    - resnet18
    - resnet34
    - resnet50
    - resnet101
    - resnet152
    - senet154
- decoder type
    - aspp

### OCNet
- encoder type
    - resnet18
    - resnet34
    - resnet50
    - resnet101
    - resnet152
    - senet154
- decoder type
    - oc_base
    - oc_aspp

## Dataset
- Cityscapes
- Pascal Voc
    - augmentation
        - http://home.bharathh.info/pubs/codes/SBD/download.html
        - https://github.com/TheLegendAli/DeepLab-Context/issues/10

## Reference

### Encoder
- https://arxiv.org/abs/1505.04597
- https://github.com/tugstugi/pytorch-saltnet

### Decoder
#### SCSE
- https://arxiv.org/abs/1803.02579

#### IBN
- https://arxiv.org/abs/1807.09441
- https://github.com/XingangPan/IBN-Net
- https://github.com/SeuTao/Kaggle_TGS2018_4th_solution

#### OC
- https://arxiv.org/abs/1809.00916
- https://github.com/PkuRainBow/OCNet

#### PSP
- https://arxiv.org/abs/1612.01105

#### ASPP
- https://arxiv.org/abs/1802.02611

### Ohter
#### inplaceABN
- https://arxiv.org/abs/1712.02616
- https://github.com/mapillary/inplace_abn
