import torch
from torch import Tensor
from torch import nn
from typing import Any, Callable, List, Optional, Type, Union
from torchvision import models

def conv3x3(in_channels: int, out_channels: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
    )

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 1, downsample = None):
        """
        Initializes the Bottleneck block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of intermediate channels.
            stride (int): Stride for the convolutional layer. Defaults to 1.
            padding (int): Padding for the convolutional layer. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer. Defaults to None.
        """
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = conv3x3(out_channels, out_channels, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True) # 直接在输入张量上进行修改，将结果存储到输入张量中
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        '''
        Defines the forward pass of the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, height, width).
        '''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 1, downsample = None):
        """
        Initializes the Bottleneck block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of intermediate channels.
            stride (int): Stride for the convolutional layer. Defaults to 1.
            padding (int): Padding for the convolutional layer. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer. Defaults to None.
        """
        super().__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(out_channels, out_channels, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        '''
        Defines the forward pass of the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, height, width).
        '''        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    '''
    Initializes the Bottleneck block. 

    Args:
        block: Basicblock or Bottleneck
        layers (list):  the list of layers number
        num_classes (int): the number of the class
    '''
    def __init__(
            self, 
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            ):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(512 * block.expansion , num_classes)

        self.layer1 = self._make_layer_(block, 64, layers[0])
        self.layer2 = self._make_layer_(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer_(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer_(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            out_channels: int,
            blocks: int,
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample=downsample)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels)
            )
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        print(f'Before Conv: {x.size()}') 
        x = self.conv1(x)
        print(f'After Conv: {x.size()}')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(f'After Maxpool: {x.size()}')

        x = self.layer1(x)
        print(f'After Layer1: {x.size()}')
        x = self.layer2(x)
        print(f'After Layer2: {x.size()}')
        x = self.layer3(x)
        print(f'After Layer3: {x.size()}')
        x = self.layer4(x)
        print(f'After Layer4: {x.size()}')

        x = self.avg_pool(x)
        print(f'After Avgpool: {x.size()}')
        x = torch.flatten(x, 1)
        print(f'After Flatten: {x.size()}')
        x = self.fc(x)
        return x

if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    # model = ResNet(Bottleneck, [3, 4, 6, 3])
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    # model2 = models.resnet18()
    print(models.mobilenet_v2())
    out = model(x)
    # print(out.size())
