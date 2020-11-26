from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet
import torch.nn as nn


from .scse import ChannelSpatialSELayer


__all__ = ['EfficientUnet', 'get_scse_eunet_b0', 'get_scse_eunet_b1', 'get_scse_eunet_b2',
           'get_scse_eunet_b3', 'get_scse_eunet_b4', 'get_scse_eunet_b5', 'get_scse_eunet_b6',
           'get_scse_eunet_b7']

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

class Attention_block(nn.Module):
    def __init__(self,in_channels_x, in_channels_g, out_channels, reduction_ratio=2):
        super(Attention_block,self).__init__()
        self.x_func = nn.Sequential(
            nn.Conv2d(in_channels_x, out_channels, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.g_func = nn.Sequential(
            nn.Conv2d(in_channels_g,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.scse = ChannelSpatialSELayer(out_channels, reduction_ratio=reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, g):
        x_1 = self.x_func(x)
        g_1 = self.g_func(g)
        output = self.relu(g_1 + x_1)
        output = self.scse(output)

        return output

class UP(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(UP, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
        self._init_weight()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
    def __init__(self, in_channels, atrous_rates, out_channels=256):
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

class doubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self._init_weight()

    def forward(self, x):
        return self.conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor=1):
        super(ResBlock, self).__init__()
        self.conv1 = doubleConv(in_channels, out_channels)
        self.conv2 = doubleConv(out_channels, in_channels//factor)
        self.conv3 = doubleConv(in_channels//factor, out_channels)
        self.elu = nn.ELU(inplace=True)
      
    def forward(self, x):
        x = self.conv1(x)
        x = self.elu(x)
        identity = x

        x = self.conv2(x)
        x = self.elu(x)
        x = self.conv3(x)

        x += identity
        x = self.elu(x)
        return x

class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2):
        super().__init__()

        self.encoder = encoder

        # decoder 
        self.up_conv1 = UP(self.n_channels, 512)
        self.atten_conv1 = Attention_block(512,self.size[0]-512,self.size[0]-512,reduction_ratio=2)
        self.double_conv1 = ResBlock(self.size[0], 512)
        
        self.up_conv2 = UP(512, 256)
        self.atten_conv2 = Attention_block(256,self.size[1]-256,self.size[1]-256,reduction_ratio=2)
        self.double_conv2 = ResBlock(self.size[1], 256)

        self.up_conv3 = UP(256, 128)
        self.atten_conv3 = Attention_block(128,self.size[2]-128,self.size[2]-128,reduction_ratio=2)
        self.double_conv3 = ResBlock(self.size[2], 128)

        self.up_conv4 = UP(128, 64)
        self.atten_conv4 = Attention_block(64,self.size[3]-64,self.size[3]-64,reduction_ratio=2)
        self.double_conv4 = ResBlock(self.size[3]+256, 64) # 256 is from aspp 

        # aspp based on output from 32x32 thus output_stride=8
        self.aspp = ASPP(self.size[1]-256, [1, 6, 12, 18])
        self.aspp_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        self.up_conv_input = UP(64, 32)
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
        _, x = blocks.popitem()

        # normal decoder part - decoder 1 
        x = self.up_conv1(x)
        g = blocks.popitem()[1] # 16x16
        atten_x = self.atten_conv1(x, g)
        x = torch.cat([x, atten_x], dim=1)
        x = self.double_conv1(x)

        # normal decoder part - decoder 2 
        x = self.up_conv2(x)
        g = blocks.popitem()[1] # 32x32
        atten_x = self.atten_conv2(x, g)
        x = torch.cat([x, atten_x], dim=1)
        x = self.double_conv2(x)

        # aspp
        aspp_x = self.aspp(g)
        aspp_x = self.aspp_up(aspp_x)

        # normal decoder part - decoder 3
        x = self.up_conv3(x)
        g = blocks.popitem()[1] # 64x64
        atten_x = self.atten_conv3(x, g)
        x = torch.cat([x, atten_x], dim=1)
        x = self.double_conv3(x)

        # normal decoder part - decoder 3
        x = self.up_conv4(x)
        g = blocks.popitem()[1] # 128x128
        atten_x = self.atten_conv4(x, g)
        x = torch.cat([x, atten_x, aspp_x], dim=1)
        x = self.double_conv4(x)

        # seg head
        x = self.up_conv_input(x)
        x = torch.cat([x, input_], dim=1)
        x = self.double_conv_input(x)

        x = self.final_conv(x)
        return x


def get_scse_eunet_b0(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_scse_eunet_b1(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_scse_eunet_b2(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_scse_eunet_b3(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_scse_eunet_b4(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_scse_eunet_b5(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model

def get_scse_eunet_b6(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model

def get_scse_eunet_b7(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model