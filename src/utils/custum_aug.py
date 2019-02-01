import random
import cv2
import numpy as np
from albumentations.core.transforms_interface import to_tuple, ImageOnlyTransform, DualTransform


def apply_motion_blur(image, count):
    """
    https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    image_t = image.copy()
    imshape = image_t.shape
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    i = imshape[1] * 3 // 4 - 10 * count
    while i <= imshape[1]:
        image_t[:, i:, :] = cv2.filter2D(image_t[:, i:, :], -1, kernel_motion_blur)
        image_t[:, :imshape[1] - i, :] = cv2.filter2D(image_t[:, :imshape[1] - i, :], -1, kernel_motion_blur)
        i += imshape[1] // 25 - count
        count += 1
    color_image = image_t
    return color_image


def rotate(img, angle, interpolation, border_mode, border_value=None):
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (width, height),
                         flags=interpolation, borderMode=border_mode, borderValue=border_value)
    return img


class AddSpeed(ImageOnlyTransform):
    def __init__(self, speed_coef=-1, p=.5):
        super().__init__(p)
        assert speed_coef == -1 or 0 <= speed_coef <= 1
        self.speed_coef = speed_coef

    def apply(self, img, count=7, **params):
        return apply_motion_blur(img, count)

    def get_params(self):
        if self.speed_coef == -1:
            return {'count': int(15 * random.uniform(0, 1))}
        else:
            return {'count': int(15 * self.speed_coef)}


class Rotate(DualTransform):
    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, border_value=255, always_apply=False, p=.5):
        super().__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value

    def apply(self, img, angle=0, **params):
        return rotate(img, angle, interpolation=self.interpolation, border_mode=self.border_mode)

    def apply_to_mask(self, img, angle=0, **params):
        return rotate(img, angle, interpolation=cv2.INTER_NEAREST,
                      border_mode=cv2.BORDER_CONSTANT, border_value=self.border_value)

    def get_params(self):
        return {'angle': random.uniform(self.limit[0], self.limit[1])}


class PadIfNeededRightBottom(DualTransform):
    def __init__(self, min_height=769, min_width=769, border_mode=cv2.BORDER_CONSTANT,
                 value=0, ignore_index=255, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value
        self.ignore_index = ignore_index

    def apply(self, img, **params):
        img_height, img_width = img.shape[:2]
        pad_height = max(0, self.min_height-img_height)
        pad_width = max(0, self.min_width-img_width)
        return np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), 'constant', constant_values=self.value)

    def apply_to_mask(self, img, **params):
        img_height, img_width = img.shape[:2]
        pad_height = max(0, self.min_height-img_height)
        pad_width = max(0, self.min_width-img_width)
        return np.pad(img, ((0, pad_height), (0, pad_width)), 'constant', constant_values=self.ignore_index)
