#!/usr/local/bin/python

import copy
import torch
import numpy as np
import torch.nn.functional as F

#from apex import amp
from torch.utils.tensorboard import SummaryWriter

from .tool import Accumulator, timer

class MultiLossEngine:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, device, num_epochs,
                scheduler=None, log_num=20, logger=None, num_class=100, name='model', apex=False):
        self.model = model 
        self.train_loader = train_loader
        self.valid_loader = valid_loader 
        self.optimizer = optimizer
        self.criterion = criterion 
        self.device = device
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.log_num = log_num
        self.logger = logger
        self.num_class = num_class 
        self.name = name
        self.apex = apex
       
        self.writer = SummaryWriter(comment=name)

        self.num_train_batch = len(self.train_loader)
        assert self.num_train_batch >= self.log_num, 'number of train batch >= log_num for logging.'
        
        self.num_valid_batch = len(self.valid_loader)
        # assert self.num_valid_batch >= self.log_num, 'number of valid batch >= log_num for logging.'

    def __repr__(self):
        '''engine info'''
        engine_info = dict()
        engine_info['name'] = self.name 
        engine_info['model'] = self.model.__class__.__name__
        engine_info['optimizer'] = self.optimizer.__class__.__name__
        engine_info['criterion'] = self.criterion.__class__.__name__
        engine_info['device'] = self.device
        return str(engine_info)

    def _forward(self, data, phase='train'):
        '''forward computation'''
        # clear parameters gradients 
        self.optimizer.zero_grad()
        
        with torch.set_grad_enabled(phase == 'train'):
            # inputs = [i.to(self.device) for i in data[:-1]]
            inputs = data[0].to(self.device)
            labels = data[-1].to(self.device)
            
            outputs = self.model(inputs)
            loss = 0
            acc = 0
            for output in outputs:
                loss += self.criterion(output, labels.long())
                acc += (output.argmax(dim=1) == labels).float().mean().item() #TODO
            self.loss = loss / len(outputs)
            avg_loss = self.loss.item()
            avg_pixel_acc = acc / len(outputs)
        
        return avg_loss, avg_pixel_acc
            
    def _backward(self):
        '''backward propagation'''
        # if self.apex:
        #     with amp.scale_loss(self.loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        # compute the gradients 
        self.loss.backward()
        # update the parameters 
        self.optimizer.step()
    
    def _initalize(self):
        '''initalize loss and acc'''
        # calculate initial loss of training dataset 
        data = iter(self.train_loader).next()
        avg_loss, avg_pixel_acc = self._forward(data, phase='valid') # this detaches grad 
        self.writer.add_scalar('avg_loss/train', avg_loss, 0)
        self.writer.add_scalar('avg_pixAcc/train', avg_pixel_acc, 0)
        
        # calculate initial loss of validation dataset 
        data = iter(self.valid_loader).next()
        avg_loss, avg_pixel_acc = self._forward(data, phase='valid') # this detaches grad 
        self.writer.add_scalar('avg_loss/valid', avg_loss, 0)
        self.writer.add_scalar('avg_pixAcc/valid', avg_pixel_acc, 0)
                        
    def _eval_step(self):
        '''validation'''
        # accumulate running loss, running true predicted, running fwiou 
        running_stat = Accumulator(3)
        valid_log_interval = 1 if self.num_valid_batch // self.log_num == 0 else self.num_valid_batch // self.log_num
        
        # set model to evaluate mode to disable batchnorm or dropout layers
        self.model.eval()
        for _ in range(valid_log_interval):
            try:
                data = next(self.valid_iter)
            except StopIteration:
                self.logger.info('Starting over on the validation set.\n')
                self.valid_iter=iter(self.valid_loader)
                data = next(self.valid_iter)
            
            # deactivate autograd engine and 
            # reduce memory usage and speed up computations when validation
            result = self._forward(data, phase='valid') # train != valid
            running_stat.add(*result)
         
        avg_loss = running_stat[0]/valid_log_interval
        avg_pixel_acc = running_stat[1]/valid_log_interval
        
        self.logger.info('[Valid] Average loss: {:.4f} Average pixAcc: {:.4f}'.format(avg_loss, avg_pixel_acc))

        self.writer.add_scalar('avg_loss/valid', avg_loss, self.step)
        self.writer.add_scalar('avg_pixAcc/valid', avg_pixel_acc, self.step)

        return avg_loss, avg_pixel_acc
    
    def train(self, metric=None, save_model=True, save_best_model=False):
        
        # record the best state
        best_acc = 0
        # best_at_epoch = 0
        # best_model_wts = copy.deepcopy(self.model.state_dict())
        # best_optimizer_wts = copy.deepcopy(self.optimizer.state_dict())
        
        self.logger.info('Engine starts......\n')
        # initialize training and validation loss 
        initalize = True 
        self.step = 0
        
        # initialize iterable validation data loader 
        self.valid_iter=iter(self.valid_loader)
        
        train_log_interval = self.num_train_batch // self.log_num
        for epoch in range(1, self.num_epochs+1):
            
            # accumulate running loss, running pixel-level acc
            running_stat = Accumulator(2)
            # epoch_f = float(epoch-1)         

            # set to training mode to enable BN and dropout 
            self.model.train()
            for batch_idx, data in enumerate(self.train_loader):
                # forward computation
                result = self._forward(data, phase='train')
                running_stat.add(*result)
                # backward propagation 
                self._backward() 
                self.step += 1
                
                if initalize:
                    self._initalize()
                    initalize = False 

                # logging
                if (batch_idx+1) % train_log_interval == 0:
                    avg_loss = running_stat[0]/train_log_interval
                    avg_pixel_acc = running_stat[1]/train_log_interval
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info('[Train] Epoch: {} [batch: {}/{} ({:05.2f}%)] Average loss: {:.4f}  Average pixAcc: {:.4f} lr: {:.10f}'.format(epoch, 
                    (batch_idx + 1), self.num_train_batch, 100.*(batch_idx + 1)/self.num_train_batch, avg_loss, avg_pixel_acc,lr))

                    self.writer.add_scalar('avg_loss/train', avg_loss, self.step)
                    self.writer.add_scalar('avg_pixAcc/train', avg_pixel_acc, self.step)
                    self.writer.add_scalar('learning rate', lr, self.step)
                    running_stat.reset()

                    # step evaluation 
                    valid_loss, valid_acc = self._eval_step()
                    
                    if valid_acc >= best_acc:
                        best_acc = valid_acc
                        self.save_state(epoch, best=True) 
                        # best_at_epoch = epoch 
                        # best_model_wts = copy.deepcopy(self.model.state_dict())
                        # best_optimizer_wts = copy.deepcopy(self.optimizer.state_dict())
                    
                    # set to training mode to enable BN and dropout 
                    self.model.train()
            
            if save_model:
                self.save_state(epoch)
                        
            # update learning rate if using scheduler
            if self.scheduler is not None:
                self.scheduler.step() 
                self.logger.info('Updating...Current learning rate: {}\n'.format(self.scheduler.get_lr()))
            
        # save the best model
        # if save_best_model:
        #     self.save_state(epoch, best=False)
            # self.model.load_state_dict(best_model_wts)
            # self.optimizer.load_state_dict(best_optimizer_wts)
            # self.save_state(best_at_epoch, best=True)

    def save_state(self, epoch, best=False):
        '''save model state'''
        filename = self.name + ('_best' if best else '') + '.pth'
        # torch.save({'epoch': epoch,
        #             'optimizer': self.optimizer.state_dict(),
        #             'state_dict': self.model.state_dict()}, 
        #             filename)
        torch.save({'state_dict': self.model.state_dict()}, 
                    filename)
        # torch.save({'state_dict': self.model.module.state_dict()}, 
        #             filename)
        self.logger.info('Saved current model as: {}'.format(filename))

    def load_state(self, state_dict_file):
        '''load model state'''
        # Open a file in read-binary mode
        with open(state_dict_file, 'rb') as f:
            self.logger.info(f"Loading state from {state_dict_file}\n")
            # interprets the file
            checkpoint = torch.load(f)
            # load network parameters
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info('Loading completed...')