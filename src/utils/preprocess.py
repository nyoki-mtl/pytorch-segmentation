import numpy as np
import cv2


def minmax_normalize(img, norm_range=(0, 1), orig_range=(0, 255)):
    # range(0, 1)
    norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
    return norm_img


def meanstd_normalize(img, mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)
    norm_img = (img - mean) / std
    return norm_img


def padding(img, pad, constant_values=0):
    pad_img = np.pad(img, pad, 'constant', constant_values=constant_values)
    return pad_img


def clahe(img, clip=2, grid=8):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    img_yuv[:, :, 0] = _clahe.apply(img_yuv[:, :, 0])
    img_equ = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2BGR)
    return img_equ
