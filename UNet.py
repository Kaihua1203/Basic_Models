import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

# Define Double Conv block(Conv 3x3)
# Notice: There is a little difference with Unet origin code -- DoubleConv doesn't change the shape of feature map
# What why we don't need padding in Up stage
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.double_conv(x)
        return out

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels :int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        out = self.maxpool_conv(x)
        return out

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        out = torch.cat([x2, x1], dim=1)
        out = self.double_conv(out)
        return out

class Unet(nn.Module):
    def __init__(self, in_channels: int, n_class: int, bilinear: bool = False):
        super().__init__()

        self.inc = DoubleConv(in_channels, out_channels=64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.out_conv = nn.Conv2d(64, n_class, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        print(f"x5 shape: {x5.shape}")

        x = self.up1(x5, x4)
        print(f"After first upsample: {x.shape}")
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.out_conv(x)
        out = self.sigmoid(out)

        return out
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    
if __name__ == '__main__':
    random_input = torch.randn(1, 3, 640, 640)
    model = Unet(3, 1, bilinear=True)
    out = model(random_input)
    print(model)




