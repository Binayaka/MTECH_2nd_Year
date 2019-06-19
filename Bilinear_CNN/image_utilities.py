"""This will hold the image resize utilities """
import os
import math
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

#seed the python environment, so as to ensure homogenous behaviour
os.environ['PYTHONHASHSEED'] = '3'
np.random.seed(3)
random.seed(3)

def resize_image(x, size_target=None, flg_keep_aspect=False, rate_scale=1.0, flg_random_scale=False):
    """This will be used to resize images """
    #convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x
    #calculate resize co-efficients
    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c = img.shape
    if len(size_target) == 1:
        size_height_target = size_target
        size_width_target = size_target
    if len(size_target) == 2:
        size_height_target = size_target[0]
        size_width_target = size_target[1]
    if size_target is None:
        size_height_target = size_height_img * rate_scale
        size_width_target = size_width_img * rate_scale

    coef_height = 1
    coef_width = 1
    if size_height_img < size_height_target:
        coef_height = size_height_target / size_height_img
    if size_width_img < size_width_target:
        coef_width = size_width_target / size_width_img

    #calculate coefficients to match small size to target size
    #scale coefficient if specified
    low_scale = rate_scale
    if flg_random_scale:
        low_scale = 1.0
    coef_max = max(coef_height, coef_width) * np.random.uniform(low=low_scale, high=rate_scale)

    #resize image
    size_height_resize = math.ceil(size_height_img * coef_max)
    size_width_resize = math.ceil(size_width_img * coef_max)

    #method interpolation = cv2.INTER_LINEAR
    #method interpolation = cv2.INTER_NEAREST
    method_interpolation = cv2.INTER_CUBIC

    if flg_keep_aspect:
        img_resized = cv2.resize(img,
                                 dsize=(size_width_resize, size_height_resize),
                                 interpolation=method_interpolation)
    else:
        img_resized = cv2.resize(img,
                                 dsize=(
                                     int(size_width_target * np.random.uniform(low=low_scale, high=rate_scale)),
                                     int(size_height_target * np.random.uniform(low=low_scale, high=rate_scale))),
                                 interpolation=method_interpolation)
    return img_resized

def resize_images(images, **kwargs):
    """This will be used to resize images as needed """
    max_images = len(images)
    new_images = []
    for i in range(max_images):
        new_images.append(resize_image(images[i], **kwargs))
    return new_images

def center_crop_image(x, size_target=(448, 448)):
    """this will crop image at the center """
    #convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x
    #set size
    if len(size_target) == 1:
        size_height_target = size_target
        size_width_target = size_target
    elif len(size_target) == 2:
        size_height_target = size_target[0]
        size_width_target = size_target[1]

    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c = img.shape

    #crop image
    h_start = int((size_height_img - size_height_target)/2)
    w_start = int((size_width_img - size_width_target)/2)

    img_cropped = img[h_start:h_start + size_height_target, w_start:w_start + size_width_target, :]
    return img_cropped

def random_crop_image(x, size_target=(448, 448)):
    """crop image of fixed size from random point of top-left corner """
    #convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x
    #set size
    if len(size_target) == 1:
        size_height_target = size_target
        size_width_target = size_target
    elif len(size_target) == 2:
        size_height_target = size_target[0]
        size_width_target = size_target[1]

    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c = img.shape
    #crop image
    margin_h = (size_height_img - size_height_target)
    margin_w = (size_width_img - size_width_target)
    h_start = 0
    w_start = 0
    if margin_h != 0:
        h_start = np.random.uniform(low=0, high=margin_h)
    if margin_w != 0:
        w_start = np.random.uniform(low=0, high=margin_w)
    img_cropped = img[h_start:h_start + size_height_target, w_start:w_start + size_width_target, :]
    return img_cropped

def horizontal_flip_image(x):
    """this will randomly flip the image """
    if np.random.random() >= 0.5:
        return x[:, ::-1, :]
    return x

def normalize_image(x, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """for feature-wise normalization"""
    x = np.asarray(x, dtype=np.float32)
    if len(x.shape) == 4:
        for dim in range(3):
            x[:, :, :, dim] = (x[:, :, :, dim] - mean[dim]) / std[dim]
    if len(x.shape) == 3:
        for dim in range(3):
            x[:, :, dim] = (x[:, :, dim] - mean[dim]) / std[dim]
    return x

def check_images(images):
    """Used to check some of the images """
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
    plt.show()
