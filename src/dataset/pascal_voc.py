from functools import partial
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu

import torch
from torch.utils.data import DataLoader, Dataset

from utils.preprocess import minmax_normalize, meanstd_normalize, padding

n_classes = 21


class PascalVocDataset(Dataset):
    def __init__(self, base_dir='../data/pascal_voc_2012/VOCdevkit/VOC2012', split="train_aug", target_size=None,
                 preprocess='imagenet', ignore_index=255, debug=False):
        self.debug = debug
        self.base_dir = Path(base_dir)
        self.ignore_index = ignore_index
        self.split = split

        valid_ids = self.base_dir.joinpath('ImageSets', 'Segmentation', 'val.txt')
        with open(valid_ids, 'r') as f:
            valid_ids = f.readlines()
        if split == 'valid':
            lbl_dir = 'SegmentationClass'
            img_ids = valid_ids
        else:
            valid_set = set([valid_id.strip() for valid_id in valid_ids])
            lbl_dir = 'SegmentationClassAug' if 'aug' in split else 'SegmentationClass'
            all_set = set([p.name[:-4] for p in self.base_dir.joinpath(lbl_dir).iterdir()])
            img_ids = list(all_set - valid_set)
        self.img_paths = [self.base_dir.joinpath('JPEGImages', f'{img_id.strip()}.jpg') for img_id in img_ids]
        self.lbl_paths = [self.base_dir.joinpath(lbl_dir, f'{img_id.strip()}.png') for img_id in img_ids]

        if target_size is None:
            self.resizer = None
        else:
            if isinstance(target_size, str):
                target_size = eval(target_size)
            self.resizer = albu.Resize(height=target_size[0], width=target_size[1])
        self.augmenter = albu.Compose([albu.HorizontalFlip(p=0.5),
                                       # albu.RandomRotate90(p=0.5),
                                       # albu.Rotate(limit=10, p=0.5),
                                       # albu.CLAHE(p=0.2),
                                       # albu.RandomContrast(p=0.2),
                                       # albu.RandomBrightness(p=0.2),
                                       # albu.RandomGamma(p=0.2),
                                       # albu.GaussNoise(p=0.2),
                                       # albu.Cutout(p=0.2)
                                       ])
        if preprocess == 'imagenet':
            self.img_preprocess = [partial(minmax_normalize, min_value=0, max_value=1),
                                   partial(meanstd_normalize, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            self.lbl_preprocess = []
        elif preprocess == 'deeplab':
            self.img_preprocess = [partial(minmax_normalize, min_value=-1, max_value=1),
                                   partial(padding, pad=((1, 2), (1, 2), (0, 0)))]
            self.lbl_preprocess = [partial(padding, pad=((0, 1), (0, 1)), constant_values=ignore_index)]
        else:
            self.img_preprocess = []
            self.lbl_preprocess = []

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        lbl_path = self.lbl_paths[index]

        img = np.array(Image.open(img_path))
        lbl = np.array(Image.open(lbl_path))
        lbl[lbl == 255] = 0

        resized = self.resizer(image=img, mask=lbl)
        img, lbl = resized['image'], resized['mask']

        if self.split != 'valid':
            augmented = self.augmenter(image=img, mask=lbl)
            img, lbl = augmented['image'], augmented['mask']

        if self.debug:
            print(np.unique(lbl))
        else:
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor(img)
            lbl = torch.LongTensor(lbl)

        return img, lbl


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dataset = PascalVocDataset('../../data/pascal_voc_2012/VOCdevkit/VOC2012/', debug=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(dataset))

    for i, batched in enumerate(dataloader):
        images, labels = batched
        if i == 0:
            fig, axes = plt.subplots(8, 2, figsize=(10, 30))
            plt.tight_layout()
            for j in range(8):
                axes[j][0].imshow(images[j])
                axes[j][1].imshow(labels[j])
            plt.savefig('pascal_voc.png')
            plt.close()
        break
