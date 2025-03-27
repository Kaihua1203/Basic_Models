import torch
from typing import Type
from torch import nn, Tensor
from torchvision import models

def conv_bn(in_channels: int, out_channels: int, stride: int, groups: int) -> nn.Sequential:
    '''Conv (kernel size 3) + BatchNorm + ReLU6'''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

def conv1x1_bn(in_channels: int, out_channels: int) -> nn.Sequential:
    '''Conv (kernel size =1) + BatchNorm + ReLU6'''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int = 6):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2], "Stride must be 1 or 2!"

        self.use_shortcut = False
        if self.stride == 1 and in_channels == out_channels:
            self.use_shortcut = True
        
        hidden_dim = int(in_channels * expand_ratio)
        
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depth-wise
                conv_bn(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # point-wise
                conv1x1_bn(hidden_dim, out_channels)
            )
        else:
            self.conv = nn.Sequential(
                # point-wise
                conv1x1_bn(in_channels, hidden_dim),
                # depth-wise
                conv_bn(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # point-wise linear
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV2(nn.Module):
    def __init__(self, 
            block: Type[InvertedResidual], 
            n_classes: int = 1000, 
            input_size: int = 224, 
            input_channels: int = 32, 
            last_channels: int = 1280,
            width_multi: float = 1.,
            ):
        super().__init__()
        
        assert input_size % 32 == 0, "Ensure that the dimensions of the input image (e.g., 224x224) are divisible by 32."

        interverted_residual_setting = [
            # t: expand_ratio,
            # c: output_channels, 
            # n: the number of reptition, 
            # s: stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.input_channels = int(input_channels * width_multi)
        self.last_channels = int(last_channels * width_multi) if width_multi > 1. else last_channels
        self.encoder = [conv_bn(3, self.input_channels, stride=2, groups=1)]

        for t, c, n, s in interverted_residual_setting:
            output_channels = int(c * width_multi)
            for i in range(n):
                if i == 0:
                    self.encoder.append(block(input_channels, output_channels, stride=s, expand_ratio=t))
                else:
                    self.encoder.append(block(input_channels, output_channels, stride=1, expand_ratio=t))
                input_channels = output_channels
        
        self.encoder.append(conv1x1_bn(input_channels, self.last_channels))
        self.encoder = nn.Sequential(*self.encoder)

        self.classifer = nn.Sequential(
            nn.Dropout(0.2), # optional
            nn.Linear(last_channels, n_classes)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert x.size(1) == 3, "The demiensions have to be 3!"
        for layer in self.encoder:
            x = layer(x)
            print(f"Layer: {layer.__class__.__name__}, Output shape: {x.shape}")
        x = self.avg_pool(x)
        print(f"After avg_pool: {x.shape}")
        x = torch.flatten(x, 1)
        print(f"After flatten: {x.shape}")
        x = self.classifer(x)
        return x

if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = MobileNetV2(InvertedResidual)
    # model = models.mobilenet_v2()
    out = model(x)
    # print(out)