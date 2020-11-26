import os 
import cv2
import torch
import numpy as np 
import ttach as tta
from torch.utils.data import Dataset, DataLoader

################################################################################
#### modified example
################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def flipud(img):
    return np.flipud(img)
    
def fliplr(img):
    return np.fliplr(img)

class mydata(Dataset):
    def __init__(self, padded_patches, crop_coord, patch_yx):
        self.padded_patches = padded_patches
        self.crop_coord = crop_coord
        self.patch_yx = patch_yx
    
    def __getitem__(self, idx):
        patch = self.padded_patches[idx]
        crop_coord = self.crop_coord[idx]
        patch_yx = self.patch_yx[idx]

        return patch, crop_coord, patch_yx

    def __len__(self):
        return len(self.padded_patches)
        
def get_img_array(img, std=False):
    img = img / 127.5 - 1 # normalized 
    img = np.transpose(img, (2,0,1)) 
    # change dimension from HxWxC to CxHxW
    if std:
        mean = np.array([-0.19934376983522895, -0.16459220809876685, -0.16546406172688],dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.17162799434448567, 0.16814594406089878, 0.1812044246744799],dtype=np.float32).reshape(3, 1, 1)
        img = (img - mean)/std 
    img = img[np.newaxis,:,:,:] # BXCxHxW
    return img

def cus_tta(img, std=False):
    img_list = []
    img_list.append(img) # 0
    img_list.append(np.rot90(img,1)) # 1
    img_list.append(np.rot90(img,2)) # 2
    img_list.append(np.rot90(img,3)) # 3
    img_list.append(fliplr(img)) # 4
    img_list.append(flipud(img)) # 5

    # part2 
    # img_list.append(fliplr(np.rot90(img,1))) #6
    # img_list.append(flipud(np.rot90(img,1))) #7
    # img_list.append(fliplr(np.rot90(img,2))) #8
    # img_list.append(flipud(np.rot90(img,2))) #9
    # img_list.append(fliplr(np.rot90(img,3))) #10
    # img_list.append(flipud(np.rot90(img,3))) #11

    img_list = np.array(img_list)
    img_list = img_list / 127.5 - 1 # normalized 
    img_list = np.transpose(img_list, (0,3,1,2)) # NxCxHxW
    if std:
        mean = np.array([-0.19934376983522895, -0.16459220809876685, -0.16546406172688],dtype=np.float32).reshape(1,3, 1, 1)
        std = np.array([0.17162799434448567, 0.16814594406089878, 0.1812044246744799],dtype=np.float32).reshape(1,3, 1, 1)
        img_list = (img_list - mean)/std 
    return img_list
    
def reverse_cus_tta(seg):
    seg_list = []
    seg = np.transpose(seg, (0,2,3,1))
    seg_list.append(seg[0])
    seg_list.append(np.rot90(seg[1],3)) 
    seg_list.append(np.rot90(seg[2],2)) 
    seg_list.append(np.rot90(seg[3],1)) 
    seg_list.append(fliplr(seg[4]))
    seg_list.append(flipud(seg[5]))

    # part2 
    # seg_list.append(np.rot90(fliplr(seg[6]),3))
    # seg_list.append(np.rot90(flipud(seg[7]),3))
    # seg_list.append(np.rot90(fliplr(seg[8]),2))
    # seg_list.append(np.rot90(flipud(seg[9]),2))
    # seg_list.append(np.rot90(fliplr(seg[10]),1))
    # seg_list.append(np.rot90(flipud(seg[11]),1))

    seg_list = np.transpose(seg_list, (0,3,1,2)) # NxCxHxW
    return seg_list

def postprocess(img):
    """
    post process image
    """
    # label-5 and label-6 does not exist in the dataset 
    # label label-5 as label-3
    img = np.where(img == 5, 3, img)

    # label label-6 as label-3
    img = np.where(img == 6, 3, img)
    
    # img = np.where(img == 4, 2, img)
    return img 

def pad_patch(patch, window_size):
    """
    """
    if patch.shape[2] == window_size[0] and patch.shape[3] == window_size[1]:
        return patch
    else:
        nrows = window_size[0] - patch.shape[2]
        ncols = window_size[1] - patch.shape[3]
        padded_patch = np.pad(patch,((0, 0), (0, 0), (0, nrows), (0, ncols)), "reflect")
        return padded_patch

def predict_sliding(model, img, std, do_tta_on_large_img, num_classes=17, overlap=0.55, window_size=(256, 256)):
    """
    """
    img = get_img_array(img, std=std) # preprocess
    _, _, img_h, img_w = img.shape

    stride = int(np.ceil(window_size[0]*(1-overlap))) # 256*0.5=128, 256*1/3=85.33, 256*0.55=140.8, 256*0.6=153.6
    stride_rows = int(np.ceil((img_h - window_size[0])/stride) + 1) # number of stride in row axis 
    stride_cols = int(np.ceil((img_w - window_size[1])/stride) + 1) # number of stride in col axis

    final_pred = np.zeros((num_classes, img_h, img_w))
    count_pred = np.zeros((img_h, img_w))

    crop_coord = []
    patch_yx = []
    padded_patches = []
    for row in range(stride_rows):
        for col in range(stride_cols):
            str_y = int(row * stride) # start y
            str_x = int(col * stride) # start x
            end_y = min(int(str_y+window_size[0]), img_h) # end y
            end_x = min(int(str_x+window_size[1]), img_w) # end x
           
            patch = img[:, :, str_y:end_y, str_x:end_x] # take a patch
            padded_patch = pad_patch(patch, window_size) # pad patch if it is smaller than window 

            crop_coord.append(np.array([str_y, end_y, str_x, end_x]))        
            patch_yx.append(np.array([patch.shape[2], patch.shape[3]]))
            padded_patch = padded_patch.squeeze()
            padded_patches.append(padded_patch)

    data = mydata(padded_patches, crop_coord, patch_yx)
    dloader = DataLoader(data, batch_size=64, num_workers=2)

    for data in dloader:
        batch_patches = data[0].cuda().float()
        batch_crop_coord = data[1]
        batch_patch_yx = data[2]
        b = len(batch_patches)

        if do_tta_on_large_img:
            transforms = tta.Compose([
                    tta.HorizontalFlip(),
                    # tta.VerticalFlip(),
                    # tta.Rotate90(angles=[0,90,180,270])
                    ])
            tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')
    
            # predict 
            with torch.no_grad():
                batch_labels = tta_model(batch_patches)
        else:
            with torch.no_grad():
                batch_labels = model(batch_patches)
        
        batch_labels = batch_labels.cpu().numpy()
        for i in range(b):
            tmp_coord = batch_crop_coord[i]
            tmp_output = batch_labels[i]
            tmp_yx = batch_patch_yx[i]
            final_pred[:, tmp_coord[0]:tmp_coord[1], tmp_coord[2]:tmp_coord[3]] += \
                tmp_output[:, 0:tmp_yx[0], 0:tmp_yx[1]] 
            count_pred[tmp_coord[0]:tmp_coord[1], tmp_coord[2]:tmp_coord[3]] += 1

    # average 
    final_pred /= count_pred 

    label = np.argmax(final_pred, axis=0).astype(np.uint8) + 1 # add 1 to label to ensure they are in [1, 17]
   
    return label

def predict_tta_one_batch(model, img, std, given_size=(256,256)):
    """
    """
    output_h, output_w = img.shape[0], img.shape[1] # height(row), width(column)

    img = cus_tta(img, std=std)

    img = torch.from_numpy(img).cuda().float() # to tensor and put it on gpu
    
    with torch.no_grad():
        label = model(img)

    label = label.cpu().numpy()
    label = reverse_cus_tta(label)
    label = np.mean(label, axis=0, keepdims=True) # merge tta by "mean"
    label = np.argmax(label, axis=1).squeeze().astype(np.uint8) + 1

    return label

def predict_by_two_way(model, img, std, do_tta_on_large_img):
    """
    """
    img_h, img_w = img.shape[0], img.shape[1]

    if img_h <= 256 and img_w <= 256: # do tta in one batch when 256 -> 1 batch 
        label = predict_tta_one_batch(model, img, std) 
    elif 256 < img_h <= 438 and 256 < img_w <= 438: # <=1 batch
        label = predict_sliding(model, img, std, do_tta_on_large_img, overlap=0.9)  
    elif 438 < img_h <= 529 and 438 < img_w <= 529: # <=1 batch
        label = predict_sliding(model, img, std, do_tta_on_large_img, overlap=0.85) 
    elif 529 < img_h <= 620 and 529 < img_w <= 620: # <=1 batch 
        label = predict_sliding(model, img, std, do_tta_on_large_img, overlap=0.8) 
    elif 620 < img_h <= 704 and 529 < img_w <= 704: # <=1 batch 
        label = predict_sliding(model, img, std, do_tta_on_large_img, overlap=0.75) 
    # elif 620 < img_h <= 1026 and 620 < img_w <= 1026: # <=2 batches
    #     label = predict_sliding(model, img, std, do_tta_on_large_img, overlap=0.7) 
    # elif 1026 < img_h <= 2056 and 1026 < img_w <= 2056: # <=7 batches
    #     label = predict_sliding(model, img, std, do_tta_on_large_img, overlap=0.65) 
    # elif 2056 < img_h <= 3758 and 2056 < img_w <= 3758: # <=20 batches
    #     label = predict_sliding(model, img, std, do_tta_on_large_img, overlap=0.6) 
    else: # <=30 batches
        label = predict_sliding(model, img, std, do_tta_on_large_img, overlap=0.55) # predict by sliding window
    return label
    
def predict(model, input_path, output_dir):
    """
    predict 
    """
    std = False # <--- change here 
    do_tta_on_large_img = False # <--- change here 

    # output name 
    name, ext = os.path.splitext(input_path) 
    name = os.path.split(name)[-1] + ".png" 

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype(np.float32) 

    # make prediction 
    label = predict_by_two_way(model, img, std, do_tta_on_large_img)
    # print('label:', label)
  
    # post process 
    label = postprocess(label)

    # save
    cv2.imwrite(os.path.join(output_dir, name), label)

