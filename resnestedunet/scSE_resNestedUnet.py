"""
SCSE + U-Net

RELU = elu
"""
import torch 
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary

# SCSE
class SCSE(nn.Module):
    def __init__(self, in_ch):
        super(SCSE, self).__init__()
        self.spatial_gate = SpatialGate2d(in_ch, 16)  # 16
        self.channel_gate = ChannelGate2d(in_ch)

    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 + g2  
        return x


# spatial squeeze and channel excitation (cSE)
class SpatialGate2d(nn.Module):
    def __init__(self, in_ch, r=16):
        super(SpatialGate2d, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        x = input_x * x

        return x

# channel squeeze and spatial excitation (sSE)
class ChannelGate2d(nn.Module):
    def __init__(self, in_ch):
        super(ChannelGate2d, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = input_x * x

        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.conv0(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.elu(out)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels//2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class scSE_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(scSE_ResBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SCSE(out_channels)

    def forward(self, x):
        identity = self.conv0(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.elu(out)
        return self.scse(out)

class scSE_ResNestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, pretrained=True, **kwargs):
        super(scSE_ResNestedUNet, self).__init__()

        self.deep_supervision = deep_supervision
        self.resnet = models.resnet34(pretrained=pretrained)
        
        num_channel = [64, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(input_channels, num_channel[0], num_channel[0]//2)
        self.conv1_0 = self.resnet.layer1 # 64
        self.conv2_0 = self.resnet.layer2
        self.conv3_0 = self.resnet.layer3
        self.conv4_0 = self.resnet.layer4

        self.conv0_1 = ResBlock(num_channel[0]+num_channel[1], num_channel[0])
        self.conv1_1 = ResBlock(num_channel[1]+num_channel[2], num_channel[1])
        self.conv2_1 = ResBlock(num_channel[2]+num_channel[3], num_channel[2])
        self.conv3_1 = scSE_ResBlock(num_channel[3]+num_channel[4], num_channel[3])

        self.conv0_2 = ResBlock(num_channel[0]*2+num_channel[1], num_channel[0])
        self.conv1_2 = ResBlock(num_channel[1]*2+num_channel[2], num_channel[1])
        self.conv2_2 = scSE_ResBlock(num_channel[2]*2+num_channel[3], num_channel[2])

        self.conv0_3 = ResBlock(num_channel[0]*3+num_channel[1], num_channel[0])
        self.conv1_3 = scSE_ResBlock(num_channel[1]*3+num_channel[2], num_channel[1])

        self.conv0_4 = scSE_ResBlock(num_channel[0]*4+num_channel[1], num_channel[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(num_channel[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(num_channel[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(num_channel[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(num_channel[0], num_classes, kernel_size=1)
            # self.final = OutConv(num_classes*4,num_classes)
            
        else:
            # self.final = nn.Conv2d(num_channel[0], num_classes, kernel_size=1)
            self.final = OutConv(num_channel[0],num_classes)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            # output = self.final(torch.cat([output1, output2, output3, output4], 1))
            return [output1, output2, output3, output4]
            # return output 
        else:
            output = self.final(x0_4)
            return output


if __name__ == "__main__":
    model = scSE_ResNestedUNet(num_classes=4, input_channels=3, pretrained=False, deep_supervision=True)
    print(summary(model, input_size=(3,256,256), device='cpu'))
