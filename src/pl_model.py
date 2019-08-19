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

import pytorch_lightning as pl

from models.net import EncoderDecoderNet, SPPNet
from losses.multi import MultiClassCriterion
from utils.optimizer import create_optimizer
from utils.metrics import compute_iou_batch


class PLModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.net_config = config['Net']
        self.data_config = config['Data']
        self.train_config = config['Train']
        self.loss_config = config['Loss']
        self.opt_config = config['Optimizer']

        self.batch_size = self.train_config['batch_size']

        # dataset
        dataset = self.data_config['dataset']
        if dataset == 'pascal':
            from dataset.pascal_voc import PascalVocDataset
            self.Dataset = PascalVocDataset
            self.net_config['output_channels'] = 21
            self.classes = np.arange(1, 21)
        elif dataset == 'cityscapes':
            from dataset.cityscapes import CityscapesDataset
            self.Dataset = CityscapesDataset
            self.net_config['output_channels'] = 19
            self.classes = np.arange(1, 19)
        else:
            raise NotImplementedError
        del self.data_config['dataset']

        # model
        if 'unet' in self.net_config['dec_type']:
            self.model = EncoderDecoderNet(**self.net_config)
        else:
            self.model = SPPNet(**self.net_config)

        # loss
        self.loss_fn = MultiClassCriterion(**self.loss_config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        images, labels, names = batch
        preds = self.model(images)
        if self.model.net_type == 'spp':
            preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=True)
        loss = self.loss_fn(preds, labels)
        iou = self.compute_metrics(preds, labels)
        return {'loss': loss, 'prog': {'tng_loss': loss, 'tng_iou': iou}}

    def validation_step(self, batch, batch_nb):
        images, labels, names = batch
        preds = self.model.tta(images)
        loss = self.loss_fn(preds, labels)
        iou = self.compute_metrics(preds, labels)
        return {'val_loss': loss, 'iou': iou}

    def compute_metrics(self, preds, labels):
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, self.classes)
        return iou

    def validation_end(self, outputs):
        valid_losses = []
        valid_ious = []
        for output in outputs:
            valid_losses.append(output['val_loss'].item())
            valid_ious.append(output['iou'])
        valid_loss = np.mean(valid_losses)
        valid_iou = np.nanmean(valid_ious)
        return {'avg_val_loss': valid_loss, 'avg_val_iou': valid_iou}

    def configure_optimizers(self):
        optimizer, scheduler = create_optimizer(self.model.parameters(), **self.opt_config)
        return [optimizer], [scheduler]

    @pl.data_loader
    def tng_dataloader(self):
        affine_augmenter = albu.Compose([albu.HorizontalFlip(p=.5),
                                         # Rotate(5, p=.5)
                                         ])
        # image_augmenter = albu.Compose([albu.GaussNoise(p=.5),
        #                                 albu.RandomBrightnessContrast(p=.5)])
        image_augmenter = None
        train_dataset = self.Dataset(affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                                     net_type=self.model.net_type, **self.data_config)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        valid_dataset = self.Dataset(split='valid', net_type=self.model.net_type, **self.data_config)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        return valid_loader
