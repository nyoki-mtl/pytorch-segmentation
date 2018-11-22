import pickle
import argparse
import yaml
import numpy as np
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.net import SegmentationNet
from logger.log import debug_logger
from logger.plot import history_ploter
from utils.optimizer import create_optimizer
from utils.metrics import compute_iou_batch
from losses.multi.ce_loss import CrossEntropy2d


def train():
    train_dataset = Dataset(split='train')
    valid_dataset = Dataset(split='valid')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SegmentationNet(**net_config).to(device)
    loss_fn = CrossEntropy2d(**loss_config).to(device)
    optimizer, scheduler = create_optimizer(model=model, **opt_config)

    if resume:
        model_path = output_dir.joinpath(f'model.pth')
        logger.info(f'Resume from {model_path}')
        param = torch.load(model_path)
        model.load_state_dict(param)
        del param

        for _ in range(start_epoch):
            scheduler.step()

        with open(log_dir.joinpath('history.pkl'), 'rb') as f:
            history_dict = pickle.load(f)
            best_metrics = history_dict['best_metrics']
            loss_history = history_dict['seg_loss']
            iou_history = history_dict['iou']
    else:
        best_metrics = 0
        loss_history = []
        iou_history = []

    for i_epoch in range(start_epoch, max_epoch):
        logger.info(f'Epoch: {i_epoch}')
        logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')

        train_losses = []
        train_ious = []
        with tqdm(train_loader) as _tqdm:
            for batched in _tqdm:
                images, labels = batched
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                preds = model(images)
                loss = loss_fn(preds, labels)

                preds_np = preds.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy().squeeze()
                iou = compute_iou_batch(preds_np, labels_np)

                _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}',
                                              iou0=f'{iou:.3f}'))
                train_losses.append(loss.item())
                train_ious.append(iou)

                loss.backward()
                optimizer.step()

        scheduler.step()

        train_loss = np.mean(train_losses)
        train_iou = np.mean(train_ious)
        logger.info(f'train loss: {train_loss}')
        logger.info(f'train iou: {train_iou}')

        valid_losses = []
        valid_ious = []
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader) as _tqdm:
                for batched in _tqdm:
                    images, labels = batched
                    images, labels = images.to(device), labels.to(device)

                    preds = model(images)
                    loss = loss_fn(preds, labels)

                    preds_np = preds.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()
                    iou = compute_iou_batch(preds_np, labels_np)

                    _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}',
                                                  iou=f'{iou:.3f}'))
                    valid_losses.append(loss.item())
                    valid_ious.append(iou)

        model.train()

        valid_loss = np.mean(valid_losses)
        valid_iou = np.mean(valid_ious)
        logger.info(f'valid seg loss: {valid_loss}')
        logger.info(f'valid iou: {valid_iou}')

        loss_history.append([train_loss, valid_loss])
        iou_history.append([train_iou, valid_iou])
        history_ploter(loss_history, log_dir.joinpath('loss.png'))
        history_ploter(iou_history, log_dir.joinpath('iou.png'))

        torch.save(model.state_dict(), output_dir.joinpath('model_tmp.pth'))
        if best_metrics < valid_iou:
            best_metrics = valid_iou
            logger.info('Best Model!')
            torch.save(model.state_dict(), output_dir.joinpath('model.pth'))

        history_dict = {'loss': loss_history,
                        'iou0': iou_history,
                        'best_metrics': best_metrics}
        with open(log_dir.joinpath('history.pkl'), 'wb') as f:
            pickle.dump(history_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    config_path = Path(args.config_path)
    config = yaml.load(open(config_path))
    net_config = config['Net']
    train_config = config['Train']
    loss_config = config['Loss']
    opt_config = config['Optimizer']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = train_config['dataset']
    if dataset == 'pascal':
        from dataset.pascal_voc import PascalVocDataset as Dataset
        net_config['output_channels'] = 21
    elif dataset == 'cityscapes':
        from dataset.cityscapes import CityscapesDataset as Dataset
        net_config['output_channels'] = 19
        loss_config['weight'] = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166,
                                                   0.9969, 0.9754, 1.0489, 0.8786, 1.0023,
                                                   0.9539, 0.9843, 1.1116, 0.9037, 1.0865,
                                                   1.0955, 1.0865, 1.1529, 1.0507]).to(device)
    else:
        raise NotImplementedError
    max_epoch = train_config['max_epoch']
    batch_size = train_config['batch_size']
    resume = train_config['resume']
    start_epoch = train_config['start_epoch'] if resume else 0

    modelname = config_path.name[:-5]
    output_dir = Path('../model').joinpath(modelname)
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path('../logs').joinpath(modelname)
    log_dir.mkdir(exist_ok=True, parents=True)

    logger = debug_logger(log_dir)
    logger.info(f'Device: {device}')
    logger.info(f'Max Epoch: {max_epoch}')

    train()
