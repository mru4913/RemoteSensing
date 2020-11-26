
#!/usr/local/bin/python

import os 
import cv2
import torch
import numpy as np

from scipy import ndimage
from skimage import transform
from skimage import exposure

MAX_IMG_VALUE = 255

def read_img_from_file(filename):
    res = []
    with open(filename, 'r') as f:
        for i in f.readlines():
            res.append(tuple(i.strip().split(',')))
    return res
    
def get_img_label_paths(images_path, labels_path):
    res = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            res.append((os.path.join(images_path, file_name+".tif"),
                        os.path.join(labels_path, file_name+".png")))
    return res

def get_item_paths(path, extension):
    res = []
    for dir_entry in os.listdir(path[0]):
        if os.path.isfile(os.path.join(path[0], dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            new = tuple()
            for p, exten in zip(path, extension):
                new += (os.path.join(p, file_name+exten),)
            res.append(new)
    return res

def get_img_path(images_path):
    res = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            res.append(os.path.join(images_path, file_name+".tif"))
    return res

def get_img_array(img):
    img = np.float32(img) / 127.5 - 1
    img = np.transpose(img, (2,0,1))
    return img
    
def get_seg_array(img):
    img = np.int32(img / 100) - 1
    img = np.float32(img)
    return img

def flipud(img):
    return np.flipud(img)
    
def fliplr(img):
    return np.fliplr(img)
    
def rotate90(img, seg):
    angle = np.random.randint(0, 4)
    img = np.rot90(img, angle)
    seg = np.rot90(seg, angle)
    return img, seg

# def gaussian_noise(img, loc=0, scale=0.1*MAX_IMG_VALUE):
#     img = img.copy().astype(np.float32)
#     noise = np.random.normal(loc, scale, img.shape[:-1]+(img.shape[-1]-1,))
#     img[..., :-1] += noise
#     img[..., :-1] = np.clip(img[..., :-1], 0, MAX_IMG_VALUE)
#     return img
    
# def gaussian_blur(img, sig_low=0.0, sig_high=3.0):
#     img = img.copy().astype(np.float32)
#     sig = np.random.uniform(sig_low, sig_high)
#     for k in range(img.shape[-1]):
#         img[..., k] = ndimage.gaussian_filter(img[..., k], sig)
#     img[..., -1] = np.clip(np.round(img[..., -1]), 0, 1)
#     img[:] = np.clip(img[:], 0, MAX_IMG_VALUE)
#     return img

def adjust_gamma(img, gamma=1.):
    return exposure.adjust_gamma(img, gamma)

def noisy(image):
    # row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[coords] = 0
    return out

def rotate(img, seg, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    seg = cv2.warpAffine(seg, rot_mat, seg.shape[1::-1], flags=cv2.INTER_LINEAR)
    return img, seg

def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def adjust_gamma(img, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)


