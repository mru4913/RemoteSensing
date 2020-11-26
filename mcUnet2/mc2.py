
# https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py

from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F

from .layers import *
from .efficientnet import EfficientNet

__all__ = ['get_mc_unet_b0','get_mc_unet_b1','get_mc_unet_b2','get_mc_unet_b3','get_mc_unet_b4', 
            'get_mc_unet_b5','get_mc_unet_b6','get_mc_unet_b7']

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

# class UP(nn.Module):
#     def __init__(self,in_channels, out_channels):
#         super(UP, self).__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ELU(inplace=True)
#         )
#         self._init_weight()

#     def forward(self, x):
#         x = self.up(x)
#         x = self.conv(x)
#         return x

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

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

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels)
#         )
#         self._init_weight()

#     def forward(self, x):
#         return self.conv(x)

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
                
class Dblock(nn.Module):
    def __init__(self,in_channels, out_channels, rate=[1,2,4]):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(in_channels, out_channels, 3, dilation=rate[0], padding=rate[0])
        self.dilate2 = nn.Conv2d(in_channels, out_channels, 3, dilation=rate[1], padding=rate[1])
        self.dilate3 = nn.Conv2d(in_channels, out_channels, 3, dilation=rate[2], padding=rate[2])
        self.nonlinearity = nn.ReLU(inplace=True)
        self._init_weight()
                    
    def forward(self, x):
        dilate1_out = self.nonlinearity(self.dilate1(x))
        dilate2_out = self.nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = self.nonlinearity(self.dilate3(dilate2_out))
        out = x + dilate1_out + dilate2_out + dilate3_out
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
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.conv0(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class D_ResBlock(nn.Module):
    """
    docstring
    """
    def __init__(self, in_channels, out_channels, rate=[1,2,4]):
        super(D_ResBlock, self).__init__()
        self.dlayer = Dblock(in_channels, in_channels, rate)
        self.resConv = ResBlock(in_channels, out_channels)

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
    def __init__(self, encoder, out_channels=2, concat=True):
        super().__init__()

        self.concat = concat
        self.encoder = encoder

        self.up = UpSample(scale_factor=2, mode='bilinear', align_corners=True)

        self.center = nn.Conv2d(self.n_channels, 512,kernel_size=1) 
       
        # self.asppConv3 = ASPP(self.size[0]-512, self.size[0]-512, [1,2,4])
        self.attenConv3 = Attention_block(512, (self.size[0]-512))
        self.resConv3 = D_ResBlock(512*2, 256)

        # self.asppConv2 = ASPP(self.size[1]-256, self.size[1]-256, [1,4,8])
        self.attenConv2 = Attention_block(256, (self.size[1]-256))
        self.resConv2 = D_ResBlock(256*2, 128)

        # self.asppConv1 = ASPP(self.size[2]-128, self.size[2]-128, [1,8,16])
        self.attenConv1 = Attention_block(128, (self.size[2]-128))
        self.resConv1 = D_ResBlock(128*2, 64)

        # self.asppConv0 = ASPP(self.size[3]-64, self.size[3]-64, [1,16,32])
        self.attenConv0 = Attention_block(64, (self.size[3]-64))
        self.resConv0 = D_ResBlock(64*2, 32)

        if self.concat:
            self.double_conv_input = ResBlock(self.size[4], 32)
        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1) 

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]


    def forward(self, x):
        input_ = x
        blocks = get_blocks_to_be_concat(self.encoder, x)

        x4 = blocks.popitem()[1]
        x3 = blocks.popitem()[1]
        x2 = blocks.popitem()[1]
        x1 = blocks.popitem()[1]
        x0 = blocks.popitem()[1]
        # x4_0 torch.Size([2, 1792, 8, 8])
        # x3_0 torch.Size([2, 112, 16, 16])
        # x2_0 torch.Size([2, 56, 32, 32])
        # x1_0 torch.Size([2, 32, 64, 64])
        # x0_0 torch.Size([2, 24, 128, 128])
        x = self.center(x4)

        # asppx3 = self.asppConv3(x3)
        # g = torch.cat([x3, asppx3], 1)
        attenx3 = self.attenConv3(self.up(x), x3) # g -> x3
        x = self.resConv3(torch.cat([self.up(x), attenx3],1))
 
        # asppx2 = self.asppConv2(x2)
        # g = torch.cat([x2, asppx2], 1)
        attenx2 = self.attenConv2(self.up(x), x2) # g -> x2 
        x = self.resConv2(torch.cat([self.up(x), attenx2],1))

        # asppx1 = self.asppConv1(x1)
        # g = torch.cat([x1, asppx1], 1)
        attenx1 = self.attenConv1(self.up(x), x1) # g -> x1
        x = self.resConv1(torch.cat([self.up(x), attenx1],1))

        # asppx0 = self.asppConv0(x0)
        # g = torch.cat([x0, asppx0], 1)
        attenx0 = self.attenConv0(self.up(x), x0) # g -> x0
        x = self.resConv0(torch.cat([self.up(x), attenx0],1))

        # seg head
        if self.concat:
            x = self.double_conv_input(torch.cat([self.up(x), input_], 1))

        x = self.final_conv(x)
        return x

def get_mc_unet_b0(out_channels=2, pretrained=True, concat=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels,concat=True)
    return model


def get_mc_unet_b1(out_channels=2, pretrained=True, concat=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat=True)
    return model

def get_mc_unet_b2(out_channels=2, pretrained=True, concat=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat=True)
    return model

def get_mc_unet_b3(out_channels=2, pretrained=True, concat=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat=True)
    return model

def get_mc_unet_b4(out_channels=2, pretrained=True, concat=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat=True)
    return model

def get_mc_unet_b5(out_channels=2, pretrained=True, concat=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat=True)
    return model

def get_mc_unet_b6(out_channels=2, pretrained=True, concat=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat=True)
    return model

def get_mc_unet_b7(out_channels=2, pretrained=True, concat=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat=True)
    return model