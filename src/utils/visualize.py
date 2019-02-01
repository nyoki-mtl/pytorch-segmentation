import numpy as np
from PIL import Image


n_classes = 5
valid_colors = [(0, 0, 0),
                (0, 0, 255),
                (255, 0, 0),
                (255, 255, 0),
                (69, 47, 142),
                ]
class_map = dict(zip(valid_colors, range(n_classes)))
own_mask = np.array(Image.open('../preprocess/own_mask010.png')).astype(bool)


def encode_mask(color_mask):
    valid_mask = np.zeros((color_mask.shape[0], color_mask.shape[1]), dtype=np.uint8)
    colors = valid_colors[1:]

    for c in colors:
        tmp_index = color_mask == c
        index = np.einsum('ij,ij,ij->ij', tmp_index[:, :, 0], tmp_index[:, :, 1], tmp_index[:, :, 2])
        valid_mask[index] = class_map[c]
    valid_mask[own_mask] = 0
    return valid_mask


def label_colormap(n=256):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3))
    for i in range(0, n):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def label2rgb(lbl, img=None, n_labels=n_classes, ignore_index=255, alpha=0.3, to_gray=False):
    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == ignore_index] = (0, 0, 0)  # unlabeled

    if img is not None:
        if to_gray:
            img = Image.fromarray(img).convert('LA')
            img = np.asarray(img.convert('RGB'))
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz
