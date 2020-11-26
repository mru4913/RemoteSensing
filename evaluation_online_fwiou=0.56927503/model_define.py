import torch
from torchvision import transforms
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet
from collections import OrderedDict
from .utils import *


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1) 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)

    # @torchsnooper.snoop()
    def forward(self, x):
        identity = self.conv0(x) 

        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.elu(out)
        return out

# attention block
# https://github.com/ztzhao6/Liver-Segmentation-with-AttentionUNet/blob/master/net_framework.py
class add_attn(nn.Module):
    def __init__(self, x_channels, g_channels=128):
        super(add_attn, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels,
                      x_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(x_channels))

        # x_channels from current stage level
        self.theta = nn.Conv2d(x_channels,
                               x_channels,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False)

        # g_channels from next stage level
        self.phi = nn.Conv2d(g_channels,
                             x_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

        self.psi = nn.Conv2d(x_channels,
                             out_channels=1,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g process
        phi_g = F.upsample(self.phi(g),
                           size=theta_x_size[2:],
                           mode='bilinear')
        # add gate
        f = F.relu(theta_x + phi_g, inplace=True)

        # roi feature
        sigm_psi_f = F.sigmoid(self.psi(f))
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear')
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y


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

# @torchsnooper.snoop()
class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=17, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input
        self.pool = nn.MaxPool2d(2, 2)
        self.deep_supervision = False
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
 
        # b3
        # num_channel = [24, 32, 48, 96, 1536]
        # b4 
        num_channel = [24, 32, 56, 112, 1792]

        self.attn_01 = add_attn(x_channels=num_channel[0], g_channels=num_channel[1])
        self.attn_02 = add_attn(x_channels=num_channel[0], g_channels=num_channel[1])
        self.attn_03 = add_attn(x_channels=num_channel[0], g_channels=num_channel[1])
        self.attn_04 = add_attn(x_channels=num_channel[0], g_channels=num_channel[1])

        self.attn_11 = add_attn(x_channels=num_channel[1], g_channels=num_channel[2])
        self.attn_12 = add_attn(x_channels=num_channel[1], g_channels=num_channel[2])
        self.attn_13 = add_attn(x_channels=num_channel[1], g_channels=num_channel[2])

        self.attn_21 = add_attn(x_channels=num_channel[2], g_channels=num_channel[3])
        self.attn_22 = add_attn(x_channels=num_channel[2], g_channels=num_channel[3])

        self.attn_31 = add_attn(x_channels=num_channel[3], g_channels=num_channel[4])

        self.conv0_1 = ResBlock(num_channel[0]+num_channel[1], num_channel[0])
        self.conv1_1 = ResBlock(num_channel[1]+num_channel[2], num_channel[1])
        self.conv2_1 = ResBlock(num_channel[2]+num_channel[3], num_channel[2])
        self.conv3_1 = ResBlock(num_channel[3]+num_channel[4], num_channel[3], dilation=4)

        self.conv0_2 = ResBlock(num_channel[0]*2+num_channel[1], num_channel[0])
        self.conv1_2 = ResBlock(num_channel[1]*2+num_channel[2], num_channel[1])
        self.conv2_2 = ResBlock(num_channel[2]*2+num_channel[3], num_channel[2], dilation=3)

        self.conv0_3 = ResBlock(num_channel[0]*3+num_channel[1], num_channel[0])
        self.conv1_3 = ResBlock(num_channel[1]*3+num_channel[2], num_channel[1], dilation=3)

        self.conv0_4 = ResBlock(num_channel[0]*4+num_channel[1], num_channel[0], dilation=2)


        if self.deep_supervision:
            self.final1 = nn.Conv2d(num_channel[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(num_channel[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(num_channel[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(num_channel[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(num_channel[0], out_channels, kernel_size=1)  

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

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


    # @torchsnooper.snoop()
    def forward(self, x):
        input_ = x
        blocks = get_blocks_to_be_concat(self.encoder, x)

        x4_0 = blocks.popitem()[1]  # (2, 1536, 16, 16)
        x3_0 = blocks.popitem()[1]  # (2, 96, 32, 32)
        x2_0 = blocks.popitem()[1]  # (2, 48, 64, 64)
        x1_0 = blocks.popitem()[1]  # (2, 32, 128, 128)
        x0_0 = blocks.popitem()[1]  # (2, 24, 256, 256)

        att0_1 = self.attn_01(x0_0, self.up(x1_0))  
        x0_1 = self.conv0_1(torch.cat([att0_1, self.up(x1_0)], 1))

        att1_1 = self.attn_11(x1_0, self.up(x2_0))
        x1_1 = self.conv1_1(torch.cat([att1_1, self.up(x2_0)], 1)) 

        att0_2 = self.attn_02(x0_1, self.up(x1_1))  
        x0_2 = self.conv0_2(torch.cat([x0_0, att0_2, self.up(x1_1)], 1))

        att2_1 = self.attn_21(x2_0, self.up(x3_0))
        x2_1 = self.conv2_1(torch.cat([att2_1, self.up(x3_0)], 1))

        att1_2 = self.attn_12(x1_1, self.up(x2_1))
        x1_2 = self.conv1_2(torch.cat([x1_0, att1_2, self.up(x2_1)], 1))

        att0_3 = self.attn_03(x0_2, self.up(x1_2))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, att0_3, self.up(x1_2)], 1))

        att3_1 = self.attn_31(x3_0, self.up(x4_0))
        x3_1 = self.conv3_1(torch.cat([att3_1, self.up(x4_0)], 1))

        att2_2 = self.attn_22(x2_1, self.up(x3_1))
        x2_2 = self.conv2_2(torch.cat([x2_0, att2_2, self.up(x3_1)], 1))

        att1_3 = self.attn_13(x1_2, self.up(x2_2))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, att1_3, self.up(x2_2)], 1))

        att0_4 = self.attn_04(x0_3, self.up(x1_3))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, att0_4, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(self.up(x0_1))
            output2 = self.final2(self.up(x0_2))
            output3 = self.final3(self.up(x0_3))
            output4 = self.final4(self.up(x0_4))
            return [output1, output2, output3, output4]
        else:
            output = self.final(self.up(x0_4))
            return output

class EfficientNet(nn.Module):

    def __init__(self, block_args_list, global_params):
        super().__init__()

        self.block_args_list = block_args_list
        self.global_params = global_params

        # Batch norm parameters
        batch_norm_momentum = 1 - self.global_params.batch_norm_momentum
        batch_norm_epsilon = self.global_params.batch_norm_epsilon

        # Stem
        in_channels = 3
        out_channels = round_filters(32, self.global_params)
        self._conv_stem = Conv2dSamePadding(in_channels,
                                            out_channels,
                                            kernel_size=3,
                                            stride=2,
                                            bias=False,
                                            name='stem_conv')
        self._bn0 = BatchNorm2d(num_features=out_channels,
                                momentum=batch_norm_momentum,
                                eps=batch_norm_epsilon,
                                name='stem_batch_norm')

        self._swish = Swish(name='swish')

        # Build _blocks
        idx = 0
        self._blocks = nn.ModuleList([])
        for block_args in self.block_args_list:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self.global_params),
                output_filters=round_filters(block_args.output_filters, self.global_params),
                num_repeat=round_repeats(block_args.num_repeat, self.global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
            idx += 1

            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=1)

            # The rest of the _blocks
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
                idx += 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self.global_params)
        self._conv_head = Conv2dSamePadding(in_channels,
                                            out_channels,
                                            kernel_size=1,
                                            bias=False,
                                            name='head_conv')
        self._bn1 = BatchNorm2d(num_features=out_channels,
                                momentum=batch_norm_momentum,
                                eps=batch_norm_epsilon,
                                name='head_batch_norm')

        # Final linear layer
        self.dropout_rate = self.global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self.global_params.num_classes)

    def forward(self, x):
        # Stem
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self._blocks)
            x = block(x, drop_connect_rate)

        # Head
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Pooling and Dropout
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Fully-connected layer
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, *, n_classes=1000, pretrained=False):
        return _get_model_by_name(model_name, classes=n_classes, pretrained=pretrained)

    @classmethod
    def encoder(cls, model_name, *, pretrained=False):
        model = cls.from_name(model_name, pretrained=pretrained)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()

                self.name = model_name

                self.global_params = model.global_params

                self.stem_conv = model._conv_stem
                self.stem_batch_norm = model._bn0
                self.stem_swish = Swish(name='stem_swish')
                self.blocks = model._blocks
                self.head_conv = model._conv_head
                self.head_batch_norm = model._bn1
                self.head_swish = Swish(name='head_swish')

            def forward(self, x):
                # Stem
                x = self.stem_conv(x)
                x = self.stem_batch_norm(x)
                x = self.stem_swish(x)

                # Blocks
                for idx, block in enumerate(self.blocks):
                    drop_connect_rate = self.global_params.drop_connect_rate
                    if drop_connect_rate:
                        drop_connect_rate *= idx / len(self.blocks)
                    x = block(x, drop_connect_rate)

                # Head
                x = self.head_conv(x)
                x = self.head_batch_norm(x)
                x = self.head_swish(x)
                return x

        return Encoder()

    @classmethod
    def custom_head(cls, model_name, *, n_classes=1000, pretrained=False):
        if n_classes == 1000:
            return cls.from_name(model_name, n_classes=n_classes, pretrained=pretrained)
        else:
            class CustomHead(nn.Module):
                def __init__(self, out_channels):
                    super().__init__()
                    self.encoder = cls.encoder(model_name, pretrained=pretrained)
                    self.custom_head = custom_head(self.n_channels * 2, out_channels)

                @property
                def n_channels(self):
                    n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                                       'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                                       'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
                    return n_channels_dict[self.encoder.name]

                def forward(self, x):
                    x = self.encoder(x)
                    mp = nn.AdaptiveMaxPool2d(output_size=(1, 1))(x)
                    ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
                    x = torch.cat([mp, ap], dim=1)
                    x = x.view(x.size(0), -1)
                    x = self.custom_head(x)

                    return x

            return CustomHead(n_classes)

def _get_model_by_name(model_name, classes=1000, pretrained=False):
    block_args_list, global_params = get_efficientnet_params(model_name, override_params={'num_classes': classes})
    model = EfficientNet(block_args_list, global_params)
    try:
        if pretrained:
            pretrained_state_dict = load_state_dict_from_url(IMAGENET_WEIGHTS[model_name])

            if classes != 1000:
                random_state_dict = model.state_dict()
                pretrained_state_dict['_fc.weight'] = random_state_dict['_fc.weight']
                pretrained_state_dict['_fc.bias'] = random_state_dict['_fc.bias']

            model.load_state_dict(pretrained_state_dict)

    except KeyError as e:
        print(f"NOTE: Currently model {e} doesn't have pretrained weights, therefore a model with randomly initialized"
              " weights is returned.")

    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model

def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model 


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model



def init_model():
    # model
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = os.path.join(os.path.dirname(__file__), 'dilate_ags_unet_008_aug_epoch220_retrain_epoch=69.pth')
    model = get_efficientunet_b4(out_channels=17, concat_input=True, pretrained=False)
    model.to(device)
    with open(model_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # evalutation model
    model.eval()
    return model