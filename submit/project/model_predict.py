import os 
import cv2
import torch
import numpy as np 
# import ttach as tta

################################################################################
#### modified example
################################################################################

def get_size(height, width):
    h = (height // 32) * 32
    w = (width // 32) * 32
    return (h, w)

def get_img_array(img):
    img = img / 127.5 - 1 # normalized 
    img = np.transpose(img, (2,0,1)) # change dimension from HxWxC to CxHxW
    img = img[np.newaxis,:,:,:] # BXCxHxW
    return img

def postprocess(img):
    """
    post process image
    """
    # label-5 and label-6 does not exist in the dataset 
    # label label-5 as label-4, coz 4 and 5 are both transportation type
    img = np.where(img == 5, 4, img)

    # label label-6 as label-8, they both belong to "others"
    img = np.where(img == 6, 8, img)
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

def predict_sliding(model, img, num_classes=17, overlap=1/3, window_size=(256, 256)):
    """
    """
    img = get_img_array(img) # preprocess
    _, _, img_h, img_w = img.shape

    stride = int(np.ceil(window_size[0]*(1-overlap)))
    stride_rows = int(np.ceil((img_h - window_size[0])/stride) + 1) # number of stride in row axis 
    stride_cols = int(np.ceil((img_w - window_size[1])/stride) + 1) # number of stride in col axis

    final_pred = np.zeros((num_classes, img_h, img_w))
    count_pred = np.zeros((img_h, img_w))

    for row in range(stride_rows):
        for col in range(stride_cols):
            str_y = int(row * stride) # start y
            str_x = int(col * stride) # start x
            end_y = min(int(str_y+window_size[0]), img_h) # end y
            end_x = min(int(str_x+window_size[1]), img_w) # end x
            #str_y = max(int(end_y-window_size[0]), 0) # double check str_y >= 0 
            #str_x = max(int(end_x-window_size[1]), 0) # double check str_x >= 0 
  
            patch = img[:, :, str_y:end_y, str_x:end_x] # take a patch
            padded_patch = pad_patch(patch, window_size) # pad patch if it is smaller than window 
            padded_patch = torch.from_numpy(padded_patch).cuda().float() # to tensor and put it on gpu

            with torch.no_grad(): # predict
                patch_output = model(padded_patch)
                
            patch_output = patch_output.squeeze().cpu().numpy()
            # fill back to the prediction of original(big) image
            # patch shape <= window_size 
            final_pred[:, str_y:end_y, str_x:end_x] += patch_output[:, 0:patch.shape[2], 0:patch.shape[3]] 
            
            # how many times each pixel is predicted 
            count_pred[str_y:end_y, str_x:end_x] += 1

    # average 
    final_pred /= count_pred 

    label = np.argmax(final_pred, axis=0).astype(np.uint8) + 1 # add 1 to label to ensure they are in [1, 17]
   
    return label

def predict_resize(model, img, given_size=(256,256)):
    """
    """
    output_h, output_w = img.shape[0], img.shape[1] # height(row), width(column)
    img = cv2.resize(img, given_size, interpolation=cv2.INTER_NEAREST) # given size (width ,height) due to cv2
    img = get_img_array(img) # preprocess
    
    img = torch.from_numpy(img).cuda().float() # to tensor and put it on gpu
    with torch.no_grad():
        label = model(img)

    label = torch.argmax(label, dim=1).cpu().squeeze().numpy().astype(np.uint8) + 1 # uint8 should also work

    label = cv2.resize(label, (output_w, output_h), interpolation=cv2.INTER_NEAREST) # resize back
    return label

def predict_by_two_way(model, img, threshold_size=(512,512)):
    """
    """
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h <= threshold_size[0] and img_w <= threshold_size[1]:
        label = predict_resize(model, img) # predict by resize
    else:
        label = predict_sliding(model, img) # predict by sliding window
    return label
    
def predict(model, input_path, output_dir):
    """
    predict 
    """
    # output name 
    name, ext = os.path.splitext(input_path) 
    name = os.path.split(name)[-1] + ".png" 

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype(np.float32) 
    # output_h, output_w = img.shape[0], img.shape[1] # height(row), width(column)
    # height, width = get_size(output_h, output_w) # **** can be removed ****
    # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST) # **** can be removed ****
    
    # make prediction 
    label = predict_by_two_way(model, img)
  
    # post process 
    label = postprocess(label)

    # resize back to the original size 
    # label = cv2.resize(label, (output_w, output_h), interpolation=cv2.INTER_NEAREST) # **** can be removed ****

    # save
    cv2.imwrite(os.path.join(output_dir, name), label)

################################################################################
#### predict
################################################################################

# def predict(model, input_path, output_dir):
#     """
#     predict 
#     """

#     # output name 
#     name, ext = os.path.splitext(input_path) 
#     name = os.path.split(name)[-1] + ".png" 

#     # read image 
#     # when you read image with cv2.IMREAD_UNCHANGED, make sure using the same way to read image 
#     # always keep the same way of reading image as what you do during training 
#     img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype(np.float32) 
#     ox, oy = img.shape[0], img.shape[1]
#     bx, by = get_size(ox, oy)
#     img = cv2.resize(img, (bx, by), interpolation=cv2.INTER_NEAREST) 
#     img = img / 127.5 - 1 # normalized 
#     

#     # to torch 
#     img = torch.from_numpy(img).unsqueeze(0) # insert new dimension 
#     img = img.permute(0, 3, 1, 2)

#     # predict 
#     with torch.no_grad():
#         label = model(img)

#     # get label 
#     label = torch.argmax(label, dim=1).cpu().squeeze().numpy().astype(np.uint16) + 1 # uint8 should also work
#     # just follow the example

#     # post process 
#     label = postprocess(label)

#     # resize back to the original size 
#     label = cv2.resize(label, (ox, oy), interpolation=cv2.INTER_NEAREST)

#     # save
#     cv2.imwrite(os.path.join(output_dir, name), label)


################################################################################
#### add tta 
################################################################################

# def predict(model, input_path, output_dir):
#     """
#     predict 
#     """

#     # output name 
#     name, ext = os.path.splitext(input_path) 
#     name = os.path.split(name)[-1] + ".png" 

#     # read image 
#     # when you read image with cv2.IMREAD_UNCHANGED, make sure using the same way to read image 
#     # always keep the same way of reading image as what you do during training 
#     img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype(np.float32) 
#     ox, oy = img.shape[0], img.shape[1]
#     bx, by = get_size(ox, oy)
#     img = cv2.resize(img, (bx, by), interpolation=cv2.INTER_NEAREST) 
#     img = img / 127.5 - 1 # normalized 

#     # to torch 
#     img = torch.from_numpy(img).unsqueeze(0) # insert new dimension 
#     img = img.permute(0, 3, 1, 2)

#     # add tta 
#     transforms = tta.Compose([
#                     # tta.HorizontalFlip(),
#                     # tta.VerticalFlip(),
#                     tta.Rotate90(angles=[0,90,180,270])
#                     ])
#     tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='max')

#     # predict 
#     with torch.no_grad():
#         label = tta_model(img)

#     # get label 
#     label = torch.argmax(label, dim=1).cpu().squeeze().numpy().astype(np.uint16) + 1 # uint8 should also work
#     # just follow the example

#     # post process 
#     label = postprocess(label)

#     # resize back to the original size 
#     label = cv2.resize(label, (ox, oy), interpolation=cv2.INTER_NEAREST)

#     # save
#     cv2.imwrite(os.path.join(output_dir, name), label)
