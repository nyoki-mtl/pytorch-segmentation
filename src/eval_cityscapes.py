import pickle
import argparse
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.net import EncoderDecoderNet, SPPNet
from dataset.cityscapes import CityscapesDataset
from utils.preprocess import minmax_normalize


parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()
config_path = Path(args.config_path)
config = yaml.load(open(config_path))
net_config = config['Net']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Network
if 'unet' in net_config['dec_type']:
    net_type = 'unet'
    model = EncoderDecoderNet(**net_config)
else:
    net_type = 'deeplab'
    # model = SPPNet(**net_config)
    model = SPPNet()
model.to(device)
model.update_bn_eps()

# modelname = config_path.stem
# model_dir = Path('../model/') / modelname
# model_path = model_dir / 'model.pth'
model_path = '../model/cityscapes_deeplab_v3_plus/model.pth'
param = torch.load(model_path)
model.load_state_dict(param)
del param

batch_size = 1
scales = [0.25, 0.75, 1, 1.25]
valid_dataset = CityscapesDataset(base_dir='/mnt/hdd0/Data/Dataset/SceneParse/CityScapes', split='valid')
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

output_dir = Path('../output/cityscapes_preds_full')
output_dir.mkdir(exist_ok=True, parents=True)

model.eval()
with torch.no_grad():
    for batched in tqdm(valid_loader):
        images, labels, names = batched
        images_np = images.numpy().transpose(0, 2, 3, 1)
        labels_np = labels.numpy()

        images, labels = images.to(device), labels.to(device)
        preds = model.tta(images, scales=scales, net_type='deeplab')
        # preds = model.pred_resize(images, images.shape[2:], net_type='deeplab')
        preds = preds.argmax(dim=1)
        preds_np = preds.detach().cpu().numpy().astype(np.uint8)

        for name, pred in zip(names, preds_np):
            Image.fromarray(pred).save(output_dir / f'{name}.png')
