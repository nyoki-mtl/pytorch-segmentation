from functools import partial
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu

import torch
from torch.utils.data import DataLoader, Dataset

from utils.preprocess import minmax_normalize, meanstd_normalize
from utils.custum_aug import PadIfNeededRightBottom


class PascalVocDataset(Dataset):
    n_classes = 21

    def __init__(self, base_dir='../data/pascal_voc_2012/VOCdevkit/VOC2012', split='train_aug',
                 affine_augmenter=None, image_augmenter=None, target_size=(512, 512),
                 net_type='unet', ignore_index=255, debug=False):
        self.debug = debug
        self.base_dir = Path(base_dir)
        assert net_type in ['unet', 'deeplab']
        self.net_type = net_type
        self.ignore_index = ignore_index
        self.split = split

        valid_ids = self.base_dir / 'ImageSets' / 'Segmentation' / 'val.txt'
        with open(valid_ids, 'r') as f:
            valid_ids = f.readlines()
        if self.split == 'valid':
            lbl_dir = 'SegmentationClass'
            img_ids = valid_ids
        else:
            valid_set = set([valid_id.strip() for valid_id in valid_ids])
            lbl_dir = 'SegmentationClassAug' if 'aug' in split else 'SegmentationClass'
            all_set = set([p.name[:-4] for p in self.base_dir.joinpath(lbl_dir).iterdir()])
            img_ids = list(all_set - valid_set)
        self.img_paths = [(self.base_dir / 'JPEGImages' / f'{img_id.strip()}.jpg') for img_id in img_ids]
        self.lbl_paths = [(self.base_dir / lbl_dir / f'{img_id.strip()}.png') for img_id in img_ids]

        # Resize
        if isinstance(target_size, str):
            target_size = eval(target_size)
        if 'train' in self.split:
            if self.net_type == 'deeplab':
                target_size = (target_size[0] + 1, target_size[1] + 1)
            self.resizer = albu.Compose([albu.RandomScale(scale_limit=(-0.5, 0.5), p=1.0),
                                         PadIfNeededRightBottom(min_height=target_size[0], min_width=target_size[1],
                                                                value=0, ignore_index=self.ignore_index, p=1.0),
                                         albu.RandomCrop(height=target_size[0], width=target_size[1], p=1.0)])
        else:
            # self.resizer = None
            self.resizer = albu.Compose([PadIfNeededRightBottom(min_height=target_size[0], min_width=target_size[1],
                                                                value=0, ignore_index=self.ignore_index, p=1.0),
                                         albu.Crop(x_min=0, x_max=target_size[1],
                                                   y_min=0, y_max=target_size[0])])

        # Augment
        if 'train' in self.split:
            self.affine_augmenter = affine_augmenter
            self.image_augmenter = image_augmenter
        else:
            self.affine_augmenter = None
            self.image_augmenter = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = np.array(Image.open(img_path))
        if self.split == 'test':
            # Resize (Scale & Pad & Crop)
            if self.net_type == 'unet':
                img = minmax_normalize(img)
                img = meanstd_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                img = minmax_normalize(img, norm_range=(-1, 1))
            if self.resizer:
                resized = self.resizer(image=img)
                img = resized['image']
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor(img)
            return img
        else:
            lbl_path = self.lbl_paths[index]
            lbl = np.array(Image.open(lbl_path))
            lbl[lbl == 255] = 0
            # ImageAugment (RandomBrightness, AddNoise...)
            if self.image_augmenter:
                augmented = self.image_augmenter(image=img)
                img = augmented['image']
            # Resize (Scale & Pad & Crop)
            if self.net_type == 'unet':
                img = minmax_normalize(img)
                img = meanstd_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                img = minmax_normalize(img, norm_range=(-1, 1))
            if self.resizer:
                resized = self.resizer(image=img, mask=lbl)
                img, lbl = resized['image'], resized['mask']
            # AffineAugment (Horizontal Flip, Rotate...)
            if self.affine_augmenter:
                augmented = self.affine_augmenter(image=img, mask=lbl)
                img, lbl = augmented['image'], augmented['mask']

            if self.debug:
                print(lbl_path)
                print(lbl.shape)
                print(np.unique(lbl))
            else:
                img = img.transpose(2, 0, 1)
                img = torch.FloatTensor(img)
                lbl = torch.LongTensor(lbl)
            return img, lbl, img_path.stem


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.custum_aug import Rotate

    affine_augmenter = albu.Compose([albu.HorizontalFlip(p=.5),
                                     Rotate(5, p=.5)
                                     ])
    # image_augmenter = albu.Compose([albu.GaussNoise(p=.5),
    #                                 albu.RandomBrightnessContrast(p=.5)])
    image_augmenter = None
    dataset = PascalVocDataset(affine_augmenter=affine_augmenter, image_augmenter=image_augmenter, split='valid',
                               net_type='deeplab', ignore_index=21, target_size=(512, 512), debug=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(dataset))

    for i, batched in enumerate(dataloader):
        images, labels, _ = batched
        if i == 0:
            fig, axes = plt.subplots(8, 2, figsize=(20, 48))
            plt.tight_layout()
            for j in range(8):
                axes[j][0].imshow(minmax_normalize(images[j], norm_range=(0, 1), orig_range=(-1, 1)))
                axes[j][1].imshow(labels[j])
                axes[j][0].set_xticks([])
                axes[j][0].set_yticks([])
                axes[j][1].set_xticks([])
                axes[j][1].set_yticks([])
            plt.savefig('dataset/pascal_voc.png')
            plt.close()
        break
