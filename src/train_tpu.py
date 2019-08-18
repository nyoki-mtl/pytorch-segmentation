import os
os.environ['XRT_TPU_CONFIG'] = 'tpu_worker;0;10.0.101.2:8470'

import pickle
import argparse
import yaml
import numpy as np
import albumentations as albu
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm

from models.net import EncoderDecoderNet, SPPNet
from losses.multi import MultiClassCriterion
from logger.log import debug_logger
from logger.plot import history_ploter
from utils.optimizer import create_optimizer
from utils.metrics import compute_iou_batch


def train_loop_fn(model, loader, device, context):
    loss_fn = MultiClassCriterion(**loss_config)
    optimizer, scheduler = create_optimizer(model.parameters(), **opt_config)

    train_losses = []
    train_ious = []
    # model.train()
    for i_iter, batched in loader:
        images, labels, names = batched
        optimizer.zero_grad()
        preds = model(images)
        if net_type == 'deeplab':
            preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=True)
        loss = loss_fn(preds, labels)

        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes)

        logger.info(f'iter: {i_iter}, seg_loss: {loss.item():.5f}, iou: {iou:.3f}')
        train_losses.append(loss.item())
        train_ious.append(iou)

        loss.backward()
        xm.optimizer_step(optimizer)

    # scheduler.step()
    return train_losses, train_ious


def valid_loop_fn(model, loader, device, context):
    valid_ious = []
    # model.eval()
    # with torch.no_grad():
    for i_iter, batched in loader:
        images, labels, names = batched
        preds = model.tta(images, net_type=net_type)

        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes)

        logger.info(f'iter: {i_iter}, iou: {iou:.3f}')
        valid_ious.append(iou)

    return valid_ious


parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()
config_path = Path(args.config_path)
config = yaml.load(open(config_path))
net_config = config['Net']
data_config = config['Data']
train_config = config['Train']
loss_config = config['Loss']
opt_config = config['Optimizer']
t_max = opt_config['t_max']

max_epoch = train_config['max_epoch']
batch_size = train_config['batch_size']
resume = train_config['resume']
pretrained_path = train_config['pretrained_path']

num_cores = None
devices = xm.get_xla_supported_devices(max_devices=num_cores)
opt_config['base_lr'] *= len(devices)

# Network
if 'unet' in net_config['dec_type']:
    net_type = 'unet'
    model = EncoderDecoderNet(**net_config)
else:
    net_type = 'deeplab'
    model = SPPNet(**net_config)

dataset = data_config['dataset']
if dataset == 'pascal':
    from dataset.pascal_voc import PascalVocDataset as Dataset
    net_config['output_channels'] = 21
    classes = np.arange(1, 21)
elif dataset == 'cityscapes':
    from dataset.cityscapes import CityscapesDataset as Dataset
    net_config['output_channels'] = 19
    classes = np.arange(1, 19)
else:
    raise NotImplementedError
del data_config['dataset']

modelname = config_path.stem
output_dir = Path('../model') / modelname
output_dir.mkdir(exist_ok=True)
log_dir = Path('../logs') / modelname
log_dir.mkdir(exist_ok=True)

logger = debug_logger(log_dir)
logger.debug(config)
logger.info(f'Device: {devices}')
logger.info(f'Max Epoch: {max_epoch}')

# history
start_epoch = 0
best_metrics = 0
loss_history = []
iou_history = []

# Dataset
affine_augmenter = albu.Compose([albu.HorizontalFlip(p=.5)])
image_augmenter = None
train_dataset = Dataset(affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                        net_type=net_type, **data_config)
valid_dataset = Dataset(split='valid', net_type=net_type, target_size=(512, 1024))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# Pretrained model
if pretrained_path:
    logger.info(f'Resume from {pretrained_path}')
    param = torch.load(pretrained_path)
    model.load_state_dict(param)
    del param

# To device
model_parallel = dp.DataParallel(model, device_ids=devices)

# Train
best_metrics = 0
for i_epoch in range(start_epoch, max_epoch):
    logger.info(f'Epoch: {i_epoch}')

    ret = model_parallel(train_loop_fn, train_loader)
    train_losses, train_ious = [], []
    for tl, ti in ret:
        train_losses += tl
        train_ious += ti
    train_loss = np.mean(train_losses)
    train_iou = np.nanmean(train_ious)
    logger.info(f'train loss: {train_loss}')
    logger.info(f'train iou: {train_iou}')

    torch.save(model_parallel._models[0].state_dict(), output_dir.joinpath('model_tmp.pth'))
    logger.info('validation start')
    if (i_epoch + 1) % 1 == 0:
        ret = model_parallel(valid_loop_fn, valid_loader)
        valid_iou = np.nanmean(ret)
        logger.info(f'valid iou: {valid_iou}')

        if best_metrics < valid_iou:
            best_metrics = valid_iou
            logger.info('Best Model!')
            torch.save(model_parallel._models[0].state_dict(), output_dir.joinpath('model.pth'))
    else:
        valid_iou = None

    loss_history.append(train_loss)
    iou_history.append([train_iou, valid_iou])
    history_ploter(loss_history, log_dir.joinpath('loss.png'))
    history_ploter(iou_history, log_dir.joinpath('iou.png'))

    history_dict = {'loss': loss_history,
                    'iou': iou_history,
                    'best_metrics': best_metrics}
    with open(log_dir.joinpath('history.pkl'), 'wb') as f:
        pickle.dump(history_dict, f)
