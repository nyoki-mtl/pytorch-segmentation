import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# class definitions -> http://apolloscape.auto/scene.html
n_classes = 36
void_classes = [0, 1, 17, 34, 162, 35, 163, 37, 165, 38, 166, 39, 167, 40, 168, 50, 65, 66, 67, 81, 82, 83, 84, 85, 86, 97, 98, 99, 100, 113, 255]
valid_classes = [33, 161, 36, 164, 49, 81]
class_map = dict(zip(valid_classes, range(n_classes)))


class ApolloscapeDataset(Dataset):
    def __init__(self,
                 base_dir='../../data/apolloscape',
                 road_record_list=[{'road':'road02_seg','record':[22, 23, 24, 25, 26]}, {'road':'road03_seg', 'record':[7, 8, 9, 10, 11, 12]}],
                 split='train',
                 ignore_index=255,
                 debug=False):
        self.debug = debug
        self.base_dir = Path(base_dir)
        self.ignore_index = ignore_index
        self.split = split
        self.img_paths = []
        self.lbl_paths = []

        for road_record in road_record_list:
          self.road_dir = self.base_dir / Path(road_record['road'])
          self.record_list = road_record['record']

          for record in self.record_list:
            img_paths_tmp = self.road_dir.glob(f'ColorImage/Record{record:03}/Camera 5/*.jpg')
            lbl_paths_tmp = self.road_dir.glob(f'Label/Record{record:03}/Camera 5/*.png')

            img_paths_basenames = {Path(img_path.name).stem for img_path in img_paths_tmp}
            lbl_paths_basenames = {Path(lbl_path.name).stem.replace('_bin', '') for lbl_path in lbl_paths_tmp}

            intersection_basenames = img_paths_basenames & lbl_paths_basenames

            img_paths_intersection = [self.road_dir / Path(f'ColorImage/Record{record:03}/Camera 5/{intersection_basename}.jpg')
                                      for intersection_basename in intersection_basenames]
            lbl_paths_intersection = [self.road_dir / Path(f'Label/Record{record:03}/Camera 5/{intersection_basename}_bin.png')
                                      for intersection_basename in intersection_basenames]

            self.img_paths += img_paths_intersection
            self.lbl_paths += lbl_paths_intersection

        self.img_paths.sort()
        self.lbl_paths.sort()
        print(len(self.img_paths), len(self.lbl_paths))
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

        if self.split == 'train':
            augmented = self.augmenter(image=img, mask=lbl)
            img, lbl = augmented['image'], augmented['mask']

        if self.debug:
            print(np.unique(lbl))
        else:
            img = self.img_transformer(img)
            lbl = self.lbl_transformer(lbl)

        return img, lbl, img_path.stem


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dataset = ApolloscapeDataset(base_dir='../../data/apolloscape', debug=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(dataset))

    for i, batched in enumerate(dataloader):
        images, labels, _ = batched
        if i == 0:
            fig, axes = plt.subplots(8, 2, figsize=(10, 30))
            plt.tight_layout()
            for j in range(8):
                axes[j][0].imshow(images[j])
                axes[j][1].imshow(labels[j])
            plt.savefig('apolloscape.png')
            plt.close()
        break
