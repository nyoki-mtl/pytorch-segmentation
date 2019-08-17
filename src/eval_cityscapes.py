import argparse
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from models.net import EncoderDecoderNet, SPPNet
from dataset.cityscapes import CityscapesDataset
from utils.preprocess import minmax_normalize

valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
id2cls_dict = dict(zip(range(19), valid_classes))
id2cls_func = np.vectorize(id2cls_dict.get)

def predict(batched, tta_flag=False):
    images, labels, names = batched
    images_np = images.numpy().transpose(0, 2, 3, 1)
    labels_np = labels.numpy()

    images, labels = images.to(device), labels.to(device)
    if tta_flag:
        preds = model.tta(images, scales=scales, net_type=net_type)
    else:
        preds = model.pred_resize(images, images.shape[2:], net_type=net_type)
    preds = preds.argmax(dim=1)
    preds_np = preds.detach().cpu().numpy().astype(np.uint8)
    return images_np, labels_np, preds_np, names

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('--tta', action='store_true')
parser.add_argument('--vis', action='store_true')
args = parser.parse_args()
config_path = Path(args.config_path)
tta_flag = args.tta
vis_flag = args.vis

config = yaml.load(open(config_path))
net_config = config['Net']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

modelname = config_path.stem
model_path = Path('../model') / modelname / 'model.pth'

if 'unet' in net_config['dec_type']:
    net_type = 'unet'
    model = EncoderDecoderNet(**net_config)
else:
    net_type = 'deeplab'
    model = SPPNet(**net_config)
model.to(device)
model.update_bn_eps()

param = torch.load(model_path)
model.load_state_dict(param)
del param

model.eval()

batch_size = 1
scales = [0.25, 0.75, 1, 1.25]
valid_dataset = CityscapesDataset(split='valid', net_type=net_type)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

if vis_flag:
    images_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for batched in valid_loader:
            images_np, labels_np, preds_np, names = predict(batched)
            images_list.append(images_np)
            labels_list.append(labels_np)
            preds_list.append(preds_np)
            if len(images_list) == 4:
                break

    images = np.concatenate(images_list)
    labels = np.concatenate(labels_list)
    preds = np.concatenate(preds_list)

    ignore_pixel = labels == 255
    preds[ignore_pixel] = 20
    labels[ignore_pixel] = 20

    fig, axes = plt.subplots(4, 3, figsize=(12, 10))
    plt.tight_layout()

    axes[0, 0].set_title('input image')
    axes[0, 1].set_title('prediction')
    axes[0, 2].set_title('ground truth')

    for ax, img, lbl, pred in zip(axes, images, labels, preds):
        ax[0].imshow(minmax_normalize(img, norm_range=(0, 1), orig_range=(-1, 1)))
        ax[1].imshow(pred)
        ax[2].imshow(lbl)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].set_xticks([])
        ax[2].set_yticks([])

    plt.savefig('eval.png')
    plt.close()
else:
    output_dir = Path('../output/cityscapes_val') / (str(modelname) + '_tta' if tta_flag else modelname)
    output_dir.mkdir(parents=True)

    with torch.no_grad():
        for batched in tqdm(valid_loader):
            _, _, preds_np, names = predict(batched)
            preds_np = id2cls_func(preds_np).astype(np.uint8)
            for name, pred in zip(names, preds_np):
                Image.fromarray(pred).save(output_dir / f'{name}.png')
