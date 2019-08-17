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

from models.net import EncoderDecoderNet, SPPNet
from losses.multi import MultiClassCriterion
from logger.log import debug_logger
from logger.plot import history_ploter
from utils.optimizer import create_optimizer
from utils.metrics import compute_iou_batch


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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
t_max = opt_config['t_max']

max_epoch = train_config['max_epoch']
batch_size = train_config['batch_size']
fp16 = train_config['fp16']
resume = train_config['resume']
pretrained_path = train_config['pretrained_path']

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
logger.info(f'Device: {device}')
logger.info(f'Max Epoch: {max_epoch}')

# Loss
loss_fn = MultiClassCriterion(**loss_config).to(device)
params = model.parameters()
optimizer, scheduler = create_optimizer(params, **opt_config)

# history
if resume:
    with open(log_dir.joinpath('history.pkl'), 'rb') as f:
        history_dict = pickle.load(f)
        best_metrics = history_dict['best_metrics']
        loss_history = history_dict['loss']
        iou_history = history_dict['iou']
        start_epoch = len(iou_history)
        for _ in range(start_epoch):
            scheduler.step()
else:
    start_epoch = 0
    best_metrics = 0
    loss_history = []
    iou_history = []

# Dataset
affine_augmenter = albu.Compose([albu.HorizontalFlip(p=.5),
                                 # Rotate(5, p=.5)
                                 ])
# image_augmenter = albu.Compose([albu.GaussNoise(p=.5),
#                                 albu.RandomBrightnessContrast(p=.5)])
image_augmenter = None
train_dataset = Dataset(affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                        net_type=net_type, **data_config)
valid_dataset = Dataset(split='valid', net_type=net_type, **data_config)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                          pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# To device
model = model.to(device)

# Pretrained model
if pretrained_path:
    logger.info(f'Resume from {pretrained_path}')
    param = torch.load(pretrained_path)
    model.load_state_dict(param)
    del param

# fp16
if fp16:
    from apex import fp16_utils
    model = fp16_utils.BN_convert_float(model.half())
    optimizer = fp16_utils.FP16_Optimizer(optimizer, verbose=False, dynamic_loss_scale=True)
    logger.info('Apply fp16')

# Restore model
if resume:
    model_path = output_dir.joinpath(f'model_tmp.pth')
    logger.info(f'Resume from {model_path}')
    param = torch.load(model_path)
    model.load_state_dict(param)
    del param
    opt_path = output_dir.joinpath(f'opt_tmp.pth')
    param = torch.load(opt_path)
    optimizer.load_state_dict(param)
    del param

# Train
for i_epoch in range(start_epoch, max_epoch):
    logger.info(f'Epoch: {i_epoch}')
    logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')

    train_losses = []
    train_ious = []
    model.train()
    with tqdm(train_loader) as _tqdm:
        for batched in _tqdm:
            images, labels, _ = batched
            if fp16:
                images = images.half()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            if net_type == 'deeplab':
                preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=True)
            if fp16:
                loss = loss_fn(preds.float(), labels)
            else:
                loss = loss_fn(preds, labels)

            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes)

            _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}', iou=f'{iou:.3f}'))
            train_losses.append(loss.item())
            train_ious.append(iou)

            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()

    scheduler.step()

    train_loss = np.mean(train_losses)
    train_iou = np.nanmean(train_ious)
    logger.info(f'train loss: {train_loss}')
    logger.info(f'train iou: {train_iou}')

    torch.save(model.state_dict(), output_dir.joinpath('model_tmp.pth'))
    torch.save(optimizer.state_dict(), output_dir.joinpath('opt_tmp.pth'))

    if (i_epoch + 1) % 1 == 0:
        valid_losses = []
        valid_ious = []
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader) as _tqdm:
                for batched in _tqdm:
                    images, labels, _ = batched
                    if fp16:
                        images = images.half()
                    images, labels = images.to(device), labels.to(device)
                    preds = model.tta(images, net_type=net_type)
                    if fp16:
                        loss = loss_fn(preds.float(), labels)
                    else:
                        loss = loss_fn(preds, labels)

                    preds_np = preds.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()
                    iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes)

                    _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}', iou=f'{iou:.3f}'))
                    valid_losses.append(loss.item())
                    valid_ious.append(iou)

        valid_loss = np.mean(valid_losses)
        valid_iou = np.mean(valid_ious)
        logger.info(f'valid seg loss: {valid_loss}')
        logger.info(f'valid iou: {valid_iou}')

        if best_metrics < valid_iou:
            best_metrics = valid_iou
            logger.info('Best Model!')
            torch.save(model.state_dict(), output_dir.joinpath('model.pth'))
            torch.save(optimizer.state_dict(), output_dir.joinpath('opt.pth'))
    else:
        valid_loss = None
        valid_iou = None

    loss_history.append([train_loss, valid_loss])
    iou_history.append([train_iou, valid_iou])
    history_ploter(loss_history, log_dir.joinpath('loss.png'))
    history_ploter(iou_history, log_dir.joinpath('iou.png'))

    history_dict = {'loss': loss_history,
                    'iou': iou_history,
                    'best_metrics': best_metrics}
    with open(log_dir.joinpath('history.pkl'), 'wb') as f:
        pickle.dump(history_dict, f)
