
# https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py

from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F

from .layers import *
from .efficientnet import EfficientNet

__all__ = ['get_mc_unet4_b0','get_mc_unet4_b1','get_mc_unet4_b2','get_mc_unet4_b3','get_mc_unet4_b4', 
            'get_mc_unet4_b5','get_mc_unet4_b6','get_mc_unet4_b7']

class Attention_block(nn.Module):
    def __init__(self, x_channels, g_channels=128):
        super(Attention_block, self).__init__()
        self.W = nn.Sequential(nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm2d(x_channels))
        self.theta = nn.Conv2d(x_channels,x_channels,kernel_size=2,stride=2,bias=False)
        self.phi = nn.Conv2d(g_channels,x_channels,kernel_size=1,stride=1,bias=True)
        self.psi = nn.Conv2d(x_channels,out_channels=1,kernel_size=1,stride=1,bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')

        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear')
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class Dblock(nn.Module):
    def __init__(self,in_channels, out_channels, rate=1, nested=True):
        super(Dblock, self).__init__()
        self.nested = nested
        # self.dilate1 = nn.Conv2d(in_channels, out_channels, 3, dilation=rate*factor, padding=rate*factor)
        # self.dilate2 = nn.Conv2d(in_channels, in_channels, 3, dilation=rate*2, padding=rate*2)
        self.nonlinearity = nn.ReLU(inplace=True)
        if nested:
            factor = 2
            self.dilate1 = nn.Conv2d(in_channels, in_channels, 3, dilation=rate*factor, padding=rate*factor)
            self.dilate2 = nn.Conv2d(in_channels, in_channels, 3, dilation=rate*2*factor, padding=rate*2*factor)
            self.conv1 = nn.Conv2d(in_channels*3, out_channels, kernel_size=1)
        else:
            factor = 1
            self.dilate1 = nn.Conv2d(in_channels, out_channels, 3, dilation=rate*factor, padding=rate*factor)
        self._init_weight()
                    
    def forward(self, x):
        if self.nested:
            x1 = self.nonlinearity(self.dilate1(x))
            x2 = self.nonlinearity(self.dilate2(x1))
            out = torch.cat([x, x1, x2], 1)
            # out = x + x1 + x2
            out = self.conv1(out)
            out = self.nonlinearity(out)
        else:
            out = self.nonlinearity(self.dilate1(x))
        return out
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self._init_weight()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class D_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rate=1, nested=True):
        super(D_ResBlock, self).__init__()
        self.dlayer = Dblock(in_channels, out_channels, rate, nested)
        self.resConv = ResBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.dlayer(x)
        x = self.resConv(x)
        return x

def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks

class UpSample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(UpSample, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = True
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, 
                        align_corners=self.align_corners)
        return x

class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2):
        super().__init__()
        self.encoder = encoder

        self.up = UpSample(scale_factor=2, mode='bilinear', align_corners=True)

        # L1 
        self.conv0_1 = D_ResBlock(self.size[0]+self.size[1], self.size[0], rate=1)

        # L2 
        self.conv1_1 = D_ResBlock(self.size[1]+self.size[2], self.size[1], rate=1)
        self.conv0_2 = D_ResBlock(self.size[0]*2+self.size[1], self.size[0], rate=2)
        
        # L3
        self.conv2_1 = D_ResBlock(self.size[2]+self.size[3], self.size[2], rate=1)
        self.conv1_2 = D_ResBlock(self.size[1]*2+self.size[2], self.size[1], rate=3)
        self.conv0_3 = D_ResBlock(self.size[0]*3+self.size[1], self.size[0], rate=4)

        # L4 
        self.conv3_1 = D_ResBlock(self.size[3]+self.size[4], self.size[3], nested=False) # <- be careful
        self.conv2_2 = D_ResBlock(self.size[2]*2+self.size[3], self.size[2])
        self.conv1_3 = D_ResBlock(self.size[1]*3+self.size[2], self.size[1])
        self.conv0_4 = D_ResBlock(self.size[0]*4+self.size[1], 32)

        # AG
        self.attenConv0 = Attention_block(self.size[0], self.size[1])
        self.attenConv1 = Attention_block(self.size[1], self.size[2])
        self.attenConv2 = Attention_block(self.size[2], self.size[3])
        self.attenConv3 = Attention_block(self.size[3], self.size[4])

        # ASPP
        self.asppConv3 = ASPP(self.size[3], self.size[3], [1,2,4,8])
        
        self.final_conv = D_ResBlock(32, 32, rate=1)
        self.seg_head = nn.Conv2d(32, out_channels, kernel_size=1) 

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [16, 24, 40, 80, 1280],
                    'efficientnet-b1': [16, 24, 40, 80, 1280],
                    'efficientnet-b2': [16, 24, 48, 88, 1408],
                    'efficientnet-b3': [24, 32, 48, 96, 1536],
                    'efficientnet-b4': [24, 32, 56, 112, 1792],
                    'efficientnet-b5': [24, 40, 64, 128, 2048],
                    'efficientnet-b6': [32, 40, 72, 144, 2304],
                    'efficientnet-b7': [32, 48, 80, 160, 2560]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        blocks = get_blocks_to_be_concat(self.encoder, x)

        x4_0 = blocks.popitem()[1]
        x3_0 = blocks.popitem()[1]
        x2_0 = blocks.popitem()[1]
        x1_0 = blocks.popitem()[1]
        x0_0 = blocks.popitem()[1]

        # L1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        # L2 
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        # L3
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        # L4 with AG
        x3_0 = self.asppConv3(x3_0)
        x3_0_atten = self.attenConv3(x3_0, self.up(x4_0))
        x3_1 = self.conv3_1(torch.cat([x3_0_atten, self.up(x4_0)], 1))

        x2_0_atten = self.attenConv2(x2_0, self.up(x3_1))
        x2_2 = self.conv2_2(torch.cat([x2_0_atten, x2_1, self.up(x3_1)], 1))

        x1_0_atten = self.attenConv1(x1_0, self.up(x2_2))
        x1_3 = self.conv1_3(torch.cat([x1_0_atten, x1_1, x1_2, self.up(x2_2)], 1))

        x0_0_atten = self.attenConv0(x0_0, self.up(x1_3))
        x0_4 = self.conv0_4(torch.cat([x0_0_atten, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x = self.final_conv(self.up(x0_4))
        x = self.seg_head(x)
        return x

def get_mc_unet4_b0(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_mc_unet4_b1(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model

def get_mc_unet4_b2(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model

def get_mc_unet4_b3(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model

def get_mc_unet4_b4(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model

def get_mc_unet4_b5(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model

def get_mc_unet4_b6(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model

def get_mc_unet4_b7(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model