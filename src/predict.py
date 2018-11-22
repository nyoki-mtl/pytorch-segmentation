import argparse
import yaml
import numpy as np
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from models.net import SegmentationNet
from utils.metrics import compute_iou_batch


def test():
    test_dataset = Dataset(split='valid')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SegmentationNet(**net_config).to(device)
    model_path = output_dir.joinpath(f'model.pth')
    param = torch.load(model_path)
    model.load_state_dict(param)
    del param

    model.eval()
    with torch.no_grad():
        for batched in tqdm(test_loader):
            images, labels = batched
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
            preds_np = np.argmax(preds.detach().cpu().numpy(), axis=1)
            labels_np = labels.detach().cpu().numpy()
            print(images_np.shape)

            iou = compute_iou_batch(preds_np, labels_np)
            print(iou)
            fig, axes = plt.subplots(batch_size, 3, figsize=(10, 4*batch_size))
            plt.tight_layout()
            for j in range(batch_size):
                axes[j][0].imshow(images_np[j])
                axes[j][1].imshow(labels_np[j])
                axes[j][2].imshow(preds_np[j])
            plt.savefig('test.png')
            plt.close()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    config_path = Path(args.config_path)
    config = yaml.load(open(config_path))
    net_config = config['Net']
    train_config = config['Train']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = train_config['dataset']
    if dataset == 'pascal':
        from dataset.pascal_voc import PascalVocDataset as Dataset
        net_config['output_channels'] = 21
    elif dataset == 'cityscapes':
        from dataset.cityscapes import CityscapesDataset as Dataset
        net_config['output_channels'] = 19
    else:
        raise NotImplementedError
    batch_size = train_config['batch_size']

    modelname = config_path.name[:-5]
    output_dir = Path('../model').joinpath(modelname)

    test()