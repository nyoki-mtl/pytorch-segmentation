import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


n_classes = 21


class PascalVocDataset(Dataset):
    def __init__(self, base_dir='../data/pascal_voc_2012/VOCdevkit/VOC2012', split="train_aug", debug=False):
        self.debug = debug
        self.base_dir = Path(base_dir)
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

        self.resizer = albu.Resize(height=256, width=256)
        self.augmenter = albu.Compose([albu.HorizontalFlip(p=0.5),
                                       # albu.RandomRotate90(p=0.5),
                                       albu.Rotate(limit=10, p=0.5),
                                       # albu.CLAHE(p=0.2),
                                       # albu.RandomContrast(p=0.2),
                                       # albu.RandomBrightness(p=0.2),
                                       # albu.RandomGamma(p=0.2),
                                       # albu.GaussNoise(p=0.2),
                                       # albu.Cutout(p=0.2)
                                       ])
        self.img_transformer = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])
        self.lbl_transformer = torch.LongTensor

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
            img = self.img_transformer(img)
            lbl = self.lbl_transformer(lbl)

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
