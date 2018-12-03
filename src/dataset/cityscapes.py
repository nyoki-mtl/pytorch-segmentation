from functools import partial
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu

import torch
from torch.utils.data import DataLoader, Dataset

from utils.preprocess import minmax_normalize, meanstd_normalize, padding

n_classes = 19
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_map = dict(zip(valid_classes, range(n_classes)))


class CityscapesDataset(Dataset):
    def __init__(self, base_dir='../data/cityscapes', split='train', target_size=None,
                 preprocess='imagenet', ignore_index=255, debug=False):
        self.debug = debug
        self.base_dir = Path(base_dir)
        self.ignore_index = ignore_index
        self.split = 'val' if split == 'valid' else split

        self.img_paths = sorted(self.base_dir.glob(f'leftImg8bit/{self.split}/*/*leftImg8bit.png'))
        self.lbl_paths = sorted(self.base_dir.glob(f'gtFine/{self.split}/*/*gtFine_labelIds.png'))
        assert len(self.img_paths) == len(self.lbl_paths)

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
        for c in void_classes:
            lbl[lbl == c] = self.ignore_index
        for c in valid_classes:
            lbl[lbl == c] = class_map[c]

        if self.resizer:
            resized = self.resizer(image=img, mask=lbl)
            img, lbl = resized['image'], resized['mask']

        if self.split != 'valid':
            augmented = self.augmenter(image=img, mask=lbl)
            img, lbl = augmented['image'], augmented['mask']

        for fn in self.img_preprocess:
            img = fn(img)
        for fn in self.lbl_preprocess:
            lbl = fn(lbl)

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

    dataset = CityscapesDataset('../../data/cityscapes', split='train', debug=True)
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
            plt.savefig('cityscapes.png')
            plt.close()
        break
