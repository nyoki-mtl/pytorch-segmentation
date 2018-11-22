# PytorchSegmentation
This repository implements general network for semantic segmentation.  
You can run various networks like UNet, PSPNet, ASPP, etc., just by writing the config file.

```
python train.py ../config/test.yaml
```

## Networks
### UNet
You can choice encoder type in  
[resnet18, resnet34, resnet50, resnet101, resnet152, resnext101_32x4d, resnext101_64x4d,  
se_resnet50, se_resnet101', se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d, senet154]  
and decoder type in  
[unet_scse, unet_seibn, unet_oc].

### PSPNet
You can choice encoder type in  
[resnet18, resnet34, resnet50, resnet101, resnet152, senet154]  
and decoder type in  
[psp].

### ASPP
You can choice encoder type in  
[resnet18, resnet34, resnet50, resnet101, resnet152, senet154]  
and decoder type in  
[aspp].

### OCNet
You can choice encoder type in  
[resnet18, resnet34, resnet50, resnet101, resnet152, senet154]  
and decoder type in  
[oc_base, oc_aspp].

## Dataset
- Cityspaces
- Pascal Voc

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
