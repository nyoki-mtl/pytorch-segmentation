import numpy as np


def minmax_normalize(img, min_value=0, max_value=1):
    # range(0, 1)
    norm_img = (img - img.min()) / (img.max() - img.min())
    # range(min_value, max_value)
    norm_img = norm_img * (max_value - min_value) + min_value
    return norm_img


def meanstd_normalize(img, mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)
    norm_img = (img - mean) / std
    return norm_img


def padding(img, pad, constant_values=0):
    pad_img = np.pad(img, pad, 'constant', constant_values=constant_values)
    return pad_img