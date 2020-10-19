"""
SCSE + U-Net
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
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
        
class expansive_block(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(expansive_block,self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.spa_cha_gate = SCSE(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.spa_cha_gate(x)
        return x
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SCSEUnet_resnet34(nn.Module):
    def __init__(self, out_channels, in_channels=3, pretrained=False, bilinear=True):
        super(SCSEUnet_resnet34, self).__init__()
        
        self.resnet = models.resnet34(pretrained=pretrained)
        
        # encoder layers 
        self.encoder0 = DoubleConv(in_channels, 64, 64//2)
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4
        
        # center layer
        self.bottleneck = nn.Sequential(DoubleConv(512, 512), SCSE(512))
        
        # decoder layers 
        self.decoder4 = expansive_block(512+512, 512, bilinear)
        self.decoder3 = expansive_block(512+256, 256, bilinear)
        self.decoder2 = expansive_block(256+128, 128, bilinear)
        self.decoder1 = expansive_block(128+64, 64, bilinear)
        self.decoder0 = DoubleConv(64, 32)
        
        # output layer
        self.output = OutConv(32, out_channels)
        
    def forward(self, x):
    
        x = self.encoder0(x)
        down1 = self.encoder1(x)
        down2 = self.encoder2(down1)
        down3 = self.encoder3(down2)
        down4 = self.encoder4(down3)
        
        bottleneck = self.bottleneck(down4)
        
        up4 = self.decoder4(bottleneck, down4)
        up3 = self.decoder3(up4, down3)
        up2 = self.decoder2(up3, down2)
        up1 = self.decoder1(up2, down1)
        up0 = self.decoder0(up1)
         
        out = self.output(up0)
        
        return out
    
if __name__ == "__main__":
    model = SCSEUnet_resnet34(in_channels=3, out_channels=4, pretrained=False)
    print(summary(model, input_size=(3,256,256), device='cpu'))
    