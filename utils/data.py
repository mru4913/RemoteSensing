import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

from utils.preprocess import *
# from utils.postprocess import *

class myDataset(Dataset):
    def __init__(self, filename, transforms=False):
        super(myDataset, self).__init__()
        self.filename = filename
        self.img_seg_pairs = read_img_from_file(filename)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_seg_pairs)
    
    def __getitem__(self, idx):
        img, seg = self.img_seg_pairs[idx]
        # print('img raw:', img)
        # print('seg raw:', seg)
        # path_251 = '/home/fc/'
        # path_253 = '/media/caixh/database/'
        # img = img.replace(path_251, path_253)
        # seg = seg.replace(path_251, path_253)

        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        seg = cv2.imread(seg, cv2.IMREAD_UNCHANGED)


        if self.transforms:
            img, seg = self.transform_fn(img, seg)
            
        img = get_img_array(img)
        seg = get_seg_array(seg)
        # print('seg:', np.where(0<=seg<=17))
        return img, seg
            
    def __repr__(self):
        return 'Dataset contains {} file sets'.format(self.__len__())
    
    def transform_fn(self, img, seg):

        angle = np.random.choice(np.arange(4))
        img, seg = np.rot90(img,angle), np.rot90(seg,angle)

        t = np.random.random()
        if 0 <= t < 0.25:
            img = fliplr(img)
            seg = fliplr(seg)
        elif 0.25 <= t < 0.5:
            img = flipud(img)
            seg = flipud(seg)  
        else:
            pass 

        if t < 0.25:
            jitter = transforms.ColorJitter(
                brightness=[0.8,1.2],
                saturation=[0.5,1.5],
                contrast=[0.5,1.5],
                hue=[-0.1,0.1])
            img = Image.fromarray(img)
            seg = Image.fromarray(seg)

            img = jitter(img)
            # img, seg = cusAffine(degrees=0,shear=[1,1.5],scale=[1,2.5])(img, seg)
            img, seg = cusAffine(degrees=0,scale=[1,2.5])(img, seg)
            img = np.array(img)
            seg = np.array(seg)

        # flag = np.random.choice(np.arange(6))
        # if flag == 0:
        #     img, seg = np.rot90(img,1), np.rot90(seg,1)
        # elif flag == 1:
        #     img, seg = np.rot90(img,2), np.rot90(seg,2)
        # elif flag == 2:
        #     img, seg = np.rot90(img,3), np.rot90(seg,3)
        # elif flag == 3:
        #     img = fliplr(img)
        #     seg = fliplr(seg)
        # elif flag == 4:
        #     img = flipud(img)
        #     seg = flipud(seg)
        # else:
        #     pass

        # gamma correction 
        # if np.random.random() < 0.5:
        #    gamma = np.random.choice(np.arange(0.80,1.35,0.05))
        #    img = adjust_gamma(img, gamma)

        # blur 
        # if np.random.random() < 0.5:
        #     img = blur(img)

        # add noise
        # if np.random.random() < 0.5:
        #     img = noisy(img)
            
        return img, seg
        
class trainDataset(Dataset):
    def __init__(self, images_path, labels_path, transforms=False):
        super(trainDataset, self).__init__()
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_seg_pairs = get_img_label_paths(images_path, labels_path)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_seg_pairs)
    
    def __getitem__(self, idx):
        img, seg = self.img_seg_pairs[idx]
        # img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        # seg = cv2.imread(seg, cv2.IMREAD_UNCHANGED)
        img = np.array(Image.open(img))
        seg = np.array(Image.open(seg))

        if self.transforms:
            img, seg = self.transform_fn(img, seg)
            
        img = get_img_array(img)
        seg = get_seg_array(seg)
        return img, seg
            
    def __repr__(self):
        return 'Dataset contains {} file sets'.format(self.__len__())
    
    def transform_fn(self, img, seg):
        # rotate by a multiple of 90
        # angle = np.random.choice(range(0,360,90))
        # img, seg = rotate(img, seg, angle)
        img, seg = rotate90(img, seg)

        # horizontal flip 
        # if np.random.random() < 0.5:
        #    img = fliplr(img)
        #    seg = fliplr(seg)

        # vertical flip
        # if np.random.random() < 0.5:
        #    img = flipud(img)
        #    seg = flipud(seg)

        # gamma correction 
        # if np.random.random() < 0.5:
        #    gamma = np.random.choice(np.arange(0.80,1.35,0.05))
        #    img = adjust_gamma(img, gamma)

        # blur 
        # if np.random.random() < 0.5:
        #     img = blur(img)

        # add noise
        # if np.random.random() < 0.5:
        #     img = noisy(img)
            
        return img, seg
        
class predictDataset(Dataset):
    def __init__(self, images_path):
        super(predictDataset, self).__init__()
        self.imgs_list = get_img_path(images_path)
        
    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, idx):
        img = self.imgs_list[idx]
        name = img
        img = np.array(Image.open(img)) # TODO
        img = get_img_array(img)
        return img,  name
            
    def __repr__(self):
        return 'Dataset contains {} file sets'.format(self.__len__())