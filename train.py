#!/usr/local/bin/python

import os
import sys 
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from net.dilate_eunet import efficientunet_resunet2_b4_AG as dilate_ags_unet

from loss import LovaszSoftmax, FocalLoss
from utils.multiLossEngine import MultiLossEngine
from utils.data import trainDataset, myDataset
from utils.engine import Engine
from utils.fengine_fwiou import finalEngine
# from utils.mengine import mixEngine
from utils.mengine_mix_loss import mixEngine
from utils.helper import str2bool

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# TODO

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
    parser.add_argument('--loss', type=str, default='lovasz', metavar='L',
                        help='loss function: ce (crossentropyloss) or lovasz (lovaszsoftmax) (default: lovasz)')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='OP',
                        help='optimizer: adam, adamw, sgd (default: adam)')
    parser.add_argument('--model', type=str, default='unet', metavar='M',
                        help='model: unet, nestedUnet, eunet(efficientUnet) (default: unet)')
    parser.add_argument('--scheduler', type=str, default='cosr', metavar='S',
                        help='scheduler: cyclic, cos (cosAnnealing), cosr (cosAnnealingWarmRestart) (default: cosr)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-6, metavar='W',
                        help='weight decay (default: 1e-6)')
    # parser.add_argument('--valid_size', type=float, default=0.15, metavar='VS',
    #                    help='valid size (default: 0.15)')
    parser.add_argument('--load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('--no_cuda', type=str2bool, default=False, nargs='?', const=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--cuda_device', type=str, default='1', nargs='?', metavar='CUDA',
                        help='cuda device number for use (default: 1)')
    parser.add_argument('--log_num', type=int, default=20, metavar='N',
                        help='the number of loggings each epoch (default: 20)')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 4)')
    parser.add_argument('--pretrained', type=str2bool, default=True, nargs='?', const=False,
                        help='pretrained')
    parser.add_argument('--deep', type=str2bool, default=False, nargs='?', const=False,
                        help='deep supervision')
    parser.add_argument('--parallel', type=str2bool, default=False, nargs='?', const=False,
                        help='parallel')
    parser.add_argument('--concat', type=str2bool, default=True, nargs='?', const=False,
                        help='concat output')
    parser.add_argument('--no_valid', type=str2bool, default=False, nargs='?', const=False,
                        help='no validation')
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
    print('filename:', filename)
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

# def get_dataloader(args):
#    data = trainDataset(args.images_path, args.labels_path,transforms=args.img_aug)
#    n_valid = int(len(data) * args.valid_size)
#   n_train = len(data) - n_valid
#   train, valid = random_split(data, [n_train, n_valid])
#   train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
#   valid_loader = DataLoader(valid, batch_size=4*args.batch_size, shuffle=True, num_workers=2)
#   return train_loader, valid_loader

def get_dataloader(args):
    train = myDataset(args.train_file_path, transforms=args.img_aug)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)

    valid = myDataset(args.valid_file_path, transforms=False) #TODO 
    valid_loader = DataLoader(valid, batch_size=10, shuffle=True, num_workers=2)
    return train_loader, valid_loader

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device #','.join(map(str,args.cuda_device))
    
    logger = get_logger(args.name)
    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # model 
    if args.model.lower() == 'dilate_ags_unet':
        model = dilate_ags_unet.get_efficientunet_b4(out_channels=args.num_class, concat_input=args.concat, deep_supervision=args.deep, 
                                    pretrained=True)    
    else:
        raise ValueError('Please select an appropriate network.')
    if args.parallel:
        print(torch.cuda.device_count())
        model.to(device)
        model = nn.DataParallel(model)
    else:
        model.to(device)

    # loss 
    if args.loss.lower() == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss.lower() == 'focal':
        criterion = FocalLoss()
    elif args.loss.lower() == 'lovasz':
        criterion = LovaszSoftmax()
    elif  args.loss.lower() == 'mix':
        criterion1 = LovaszSoftmax()
        criterion2 = nn.CrossEntropyLoss()
    elif args.loss.lower() == 'multi':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Please select an appropriate loss function.')

    # optimizer     
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, 
            momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, 
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, 
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    else:
        scheduler = None 
    
    # data loader 
    train_loader, valid_loader = get_dataloader(args)
    # TODO
    if args.loss.lower() == 'mix':
        engine = mixEngine(model, train_loader, valid_loader, optimizer, criterion1, criterion2, device, 
        num_epochs=args.num_epochs, scheduler=scheduler, log_num=args.log_num, logger=logger,
        num_class=args.num_class, name=args.name) 
    elif args.loss.lower() == 'multi':
        engine = MultiLossEngine(model, train_loader, valid_loader, optimizer, criterion, device, 
        num_epochs=args.num_epochs, scheduler=scheduler, log_num=args.log_num, logger=logger,
        num_class=args.num_class, name=args.name)
    else:
        engine = finalEngine(model, train_loader, valid_loader, optimizer, criterion, device, 
        num_epochs=args.num_epochs, scheduler=scheduler, log_num=args.log_num, logger=logger,
        num_class=args.num_class, name=args.name) 

    # load state
    if args.load:
        engine.load_state(args.load)
        
    try:
        engine.train(metric=None, save_model=True, save_best_model=True)
    except KeyboardInterrupt:
        torch.save({'state_dict': engine.model.state_dict()}, 'INTERRUPTED_'+args.name+'.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
  