#!/usr/local/bin/python

import os
import sys 
import logging
import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

# student model 
from student_model import UNet
# teacher 
from mcUnet3 import get_mc_unet3_b0, get_mc_unet3_b4, get_mc_unet3_b5, get_mc_unet3_b6, get_mc_unet3_b7

from utils.data import myDataset
from utils.distillEngine import DistillEngine
from utils.helper import str2bool

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

def parse_args():
    '''
    Parse input arguments from users
    '''
    # add descriptions 
    parser = argparse.ArgumentParser(description='segmentation')
    
    # add user arguments
    parser.add_argument('train_file_path', type=str, 
                        help='the file path for training')
    parser.add_argument('valid_file_path', type=str, 
                        help='the file path for validation')
    parser.add_argument('--name', type=str, default='myUnet',
                        help='a name for this training')
    parser.add_argument('--num_class', type=int, default=17, metavar='N',
                        help='number of class (default: 17)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=2, metavar='N',
                        help='the number of epochs for training (default: 2)')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='OP',
                        help='optimizer: adam, adamw, sgd (default: adam)')
    parser.add_argument('--teacher_model', type=str, default='mc3', metavar='M',
                        help='teacher model selection')
    parser.add_argument('--load_teacher_model', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('--student_model', type=str, default='unet', metavar='M',
                        help='student model selection')
    parser.add_argument('--t', type=int, default=5, metavar='M',
                        help='temperature (default: 5)')
    parser.add_argument('--alpha', type=float, default=0.5, metavar='M',
                        help='alpha 0~1 (default: 0.5)')
    parser.add_argument('--scheduler', type=str, default='cosr', metavar='S',
                        help='scheduler: cyclic, cos (cosAnnealing), cosr (cosAnnealingWarmRestart) (default: cosr)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='W',
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--no_cuda', type=str2bool, default=False, nargs='?', const=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--cuda_device', type=int, default=1, nargs='?', metavar='CUDA',
                        help='cuda device number for use (default: 1)')
    parser.add_argument('--log_num', type=int, default=20, metavar='N',
                        help='the number of loggings each epoch (default: 20)')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 4)')
    parser.add_argument('--pretrained', type=str2bool, default=False, nargs='?', const=False,
                        help='pretrained')
    parser.add_argument('--deep', type=str2bool, default=False, nargs='?', const=False,
                        help='deep supervision')
    parser.add_argument('--concat', type=str2bool, default=False, nargs='?', const=False,
                        help='concat output')
    parser.add_argument('--img_aug', type=str2bool, default=False, nargs='?', const=False,
                        help='using image augmentation (default: False)')
    parser.add_argument('--save_model', type=str2bool, default=False, nargs='?', const=False,
                        help='saving the optimal Model during training (default: False)')
    args = parser.parse_args()

    return args

def get_logger(filename, path='./log'):
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
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
    formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(message)s")  #定义该handler格式
    fHandler.setFormatter(formatter)

    logger.addHandler(fHandler)
#    logger.addHandler(cHandler)

    logger.info("Start logging...")
    return logger 

def get_dataloader(args):
    train = myDataset(args.train_file_path, transforms=args.img_aug)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    valid = myDataset(args.valid_file_path, transforms=False) 
    valid_loader = DataLoader(valid, batch_size=32, shuffle=True, num_workers=4)
    return train_loader, valid_loader

def load_state(model, device, state_dict_file):
    with open(state_dict_file, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model 

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    
    logger = get_logger(args.name)
    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # teacher model 
    if args.teacher_model.lower() == 'mc3':
        teacher_model = get_mc_unet3_b4(out_channels=args.num_class,pretrained=False)
    else:
        raise ValueError('Please select an appropriate network.')
    teacher_model.to(device)
    teacher_model = load_state(teacher_model, device, args.load_teacher_model) # load pre-trained teacher model 
    teacher_model = teacher_model.eval() # evaluation only 

    # student model 
    if args.student_model.lower() == 'unet':
        student_model = UNet(n_class=args.num_class)
    else:
        raise ValueError('Please select an appropriate network.')
    student_model.to(device)
    
    # loss 
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.KLDivLoss() 

    # optimizer   
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, student_model.parameters()), lr=args.lr, 
            momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, student_model.parameters()), lr=args.lr, 
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, student_model.parameters()), lr=args.lr, 
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        raise ValueError('Please select an appropriate optimizer.')

    # scheduler
    if args.scheduler.lower() == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=1e-2)
    elif args.scheduler.lower() == 'cosr':    
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
    elif args.scheduler.lower() == 'cos':
        assert args.num_epochs >= 10, "Make sure you have big enough num_epochs when using CosAnnealing scheduler"
        T_max = args.num_epochs // 5  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.lr)
    elif args.scheduler.lower() == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.scheduler.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    else:
        scheduler = None 

    # data loader 
    train_loader, valid_loader = get_dataloader(args)

    engine = DistillEngine(student_model, teacher_model, train_loader, valid_loader, optimizer, criterion1, 
            criterion2,device, num_epochs=args.num_epochs, scheduler=scheduler, log_num=args.log_num, logger=logger,
        num_class=args.num_class, name=args.name, t=args.t, alpha=args.alpha) 

    try:
        engine.train(metric=None, save_model=True, save_best_model=True)
    except KeyboardInterrupt:
        torch.save({'state_dict': engine.student_model.state_dict()}, 'INTERRUPTED_'+args.name+'.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == "__main__":
    args = parse_args()
    main(args)



