import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


n_classes = 19
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_map = dict(zip(valid_classes, range(n_classes)))


class CityscapesDataset(Dataset):
    def __init__(self, base_dir='../data/cityscapes', split="train", ignore_index=255, debug=False):
        self.debug = debug
        self.base_dir = Path(base_dir)
        self.ignore_index = ignore_index
        self.split = split

        self.img_paths = sorted(self.base_dir.glob(f'leftImg8bit/{split}/*/*leftImg8bit.png'))
        self.lbl_paths = sorted(self.base_dir.glob(f'gtFine/{split}/*/*gtFine_labelIds.png'))
        assert len(self.img_paths) == len(self.lbl_paths)

        self.resizer = albu.Resize(height=512, width=1024)
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
        for c in void_classes:
            lbl[lbl == c] = self.ignore_index
        for c in valid_classes:
            lbl[lbl == c] = class_map[c]

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

    dataset = CityscapesDataset('../../data/cityscapes', debug=True)
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