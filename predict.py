import os 
import argparse
import logging 

import torch
import numpy as np 
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from unet import UNet
from nestedunet import NestedUNet
from scse_eunet import get_scse_eunet_b4, get_scse_eunet_b5
from efficientunet import get_efficientunet_b5
from resnestedunet import ResNestedUNet, scSE_ResNestedUNet
from scSE_unet_resnet34 import SCSEUnet_resnet34

from utils.data import predictDataset
from utils.helper import str2bool

def parse_args():
    '''
    Parse input arguments from users
    '''
    # add descriptions 
    parser = argparse.ArgumentParser(description='segmentation')
    
    # add user arguments
    parser.add_argument('images_path', type=str, 
                        help='image path')
    parser.add_argument('save_path', type=str, 
                        help='save path')
    parser.add_argument('--name', type=str, default='hist',
                        help='name')
    parser.add_argument('--num_class', type=int, default=8, metavar='N',
                        help='number of class (default: 8)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--model', type=str, default='unet', metavar='M',
                        help='model: unet, nestedUnet, eunet(efficientUnet) (default: unet)')
    parser.add_argument('--load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('--no_cuda', type=str2bool, default=False, nargs='?', const=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--cuda_device', type=int, default=1, metavar='CUDA',
                        help='cuda device number for use (default: 1)')
    parser.add_argument('--concat', type=str2bool, default=True, nargs='?', const=False,
                        help='concat output')
    parser.add_argument('--tta', type=str2bool, default=False, nargs='?', const=False,
                        help='TTA')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 4)')
    args = parser.parse_args()

    return args

def check_image(img):
    arr = np.unique(img)
    if not np.logical_and(arr>=100, arr<=800).all():
        return True 

def save_image(img, img_name):
    Image.fromarray(img).save(img_name) 

def postprocess(imgs, img_names, logger, path):
    imgs = (np.int32(imgs) + 1) * 100 
    for i, img_name in enumerate(img_names):
        img_name = os.path.splitext(img_name.split('/')[-1])[0]
        img_name = os.path.join(path, img_name + '.png')
        if check_image(imgs[i]):
          logger.info(f'Fail to generate {img_name}')
        save_image(imgs[i], img_name)

def predict(model, test_loader, device, logger, path='/predict'):
    logger.info('Start evaluating...')
    # set model to evaluate model
    model.eval()
    # deactivate autograd engine and 
    # reduce memory usage and speed up computations
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs = [i.to(device) for i in data[:-1]]
            outputs = model(*inputs)
            outputs = outputs.argmax(dim=1).cpu().numpy()
            image_names = data[-1]
            postprocess(outputs, image_names, logger, path)

def predict_tta(model, test_loader, device, logger, path='/predict'):
    import ttach as tta
    logger.info('Start evaluating...')
    # set model to evaluate model
    model.eval()

    transforms = tta.Compose([tta.HorizontalFlip(),
                     tta.VerticalFlip(),
                      tta.Rotate90(angles=[0,90,180,270])])
    # tta.aliases.d4_transform()
    tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='max')
    # deactivate autograd engine and 
    # reduce memory usage and speed up computations
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs = [i.to(device) for i in data[:-1]]
            outputs = tta_model(*inputs)
            outputs = outputs.argmax(dim=1).cpu().numpy()
            image_names = data[-1]
            postprocess(outputs, image_names, logger, path)

def load_state(model, device, state_dict_file):
    with open(state_dict_file, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
 
def get_logger(filename, path='./'):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S") 
    logger = logging.getLogger(filename)
    
#    cHandler = logging.StreamHandler()
#    cHandler.setLevel(logging.INFO)

    filename = os.path.join(path, filename+'.log')
    try:
        fHandler = logging.FileHandler(filename)
    except FileNotFoundError:
        os.mknod(filename)
        fHandler = logging.FileHandler(filename)
    fHandler.setLevel(logging.INFO)

    logger.addHandler(fHandler)
#    logger.addHandler(cHandler)

    logger.info("Start logging...")
    return logger 

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    
    logger = get_logger(args.name)
    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.model.lower() == 'unet':
        model = UNet(n_channels=3, n_classes=args.num_class, bilinear=True)
    elif args.model.lower() == 'nestedunet':
        model = NestedUNet(num_classes=args.num_class, input_channels=3)
    elif args.model.lower() == 'eunet':
        model = get_scse_eunet_b4(out_channels=args.num_class, concat_input=args.concat)
    elif args.model.lower() == 'eunet_b5':
        model = get_scse_eunet_b5(out_channels=args.num_class, concat_input=args.concat)
    elif args.model.lower() == 'basic_eunet':
        model = get_efficientunet_b5(out_channels=args.num_class, concat_input=args.concat)
    elif args.model.lower() == 'res':
        model = ResNestedUNet(num_classes=args.num_class)
    elif args.model.lower() == 'scseunet':
        model = SCSEUnet_resnet34(out_channels=args.num_class)
    elif args.model.lower() == 'scse_nestedunet':
        model = scSE_ResNestedUNet(num_classes=args.num_class)
    else:
        raise ValueError('Please select an appropriate network.')
    model.to(device)

    logger.info(f'loading {args.load}')
    load_state(model, device, args.load)

    # data 
    mydataset = predictDataset(args.images_path)
    mydataloader = DataLoader(mydataset, batch_size=args.batch_size, num_workers=4) 

    if args.tta:
        print("Using TTA")
        predict_tta(model, mydataloader, device, logger, args.save_path)
    else:
        predict(model, mydataloader, device, logger, args.save_path)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
