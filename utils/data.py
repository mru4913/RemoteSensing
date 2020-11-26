import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms
from utils.preprocess import *

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
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        seg = cv2.imread(seg, cv2.IMREAD_UNCHANGED)

        if self.transforms:
            img, seg = self.transform_fn(img, seg)
            
        img = get_img_array(img)
        seg = get_seg_array(seg)
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
            # jitter = transforms.ColorJitter( #mc3s use this 
            #     brightness=[0.8,1.2],
            #     saturation=[0.5,1.5],
            #     contrast=[0.5,1.5],
            #     hue=[-0.1,0.1])
            jitter = transforms.ColorJitter( # mc7 use this 
                brightness=[0.8,1.2],
                saturation=[0.8,1.2],
                contrast=[0.8,1.2],
                hue=[-0.1,0.1])
            img = Image.fromarray(img)
            seg = Image.fromarray(seg)
            img = jitter(img)
            img, seg = cusAffine(degrees=0,scale=[1,2])(img, seg) # 2.5 -> 2
            img = np.array(img)
            seg = np.array(seg)

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
        # img = np.array(Image.open(img))
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        img = get_img_array(img)
        return img,  name
            
    def __repr__(self):
        return 'Dataset contains {} file sets'.format(self.__len__())