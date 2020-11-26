from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet
import torchsnooper


__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


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

# extention conv block
class Dila_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(Dila_ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0, dilation=dilation)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.resBlock = ResBlock(in_channels, out_channels)
    
    # @torchsnooper.snoop()
    def forward(self, x):
        out = self.conv(x)  
        out = self.bn(out)
        out = self.relu(out)
        out = self.resBlock(out)
        return out


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
    def __init__(self, encoder, out_channels=17, concat_input=True, deep_supervision=False):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input
        self.pool = nn.MaxPool2d(2, 2)
        self.deep_supervision = deep_supervision
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        num_channel = self.size
        # b3
        # num_channel = [24, 32, 48, 96, 1536]
        # b4 
        # num_channel = [24, 32, 56, 112, 1792]

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


    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    # @property
    # def size(self):
    #     size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
    #                  'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
    #                  'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
    #                  'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
    #     return size_dict[self.encoder.name]

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


def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, deep_supervision=False, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, deep_supervision=deep_supervision, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, deep_supervision=False, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, deep_supervision=deep_supervision, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, deep_supervision=False, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, deep_supervision=deep_supervision, concat_input=concat_input)
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



if __name__ == "__main__":
    pass