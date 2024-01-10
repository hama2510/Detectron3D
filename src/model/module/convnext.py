import torch
from collections import OrderedDict


class SeparableConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(SeparableConv, self).__init__()
        padding = kernel_size//2
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ConvNext(torch.nn.Module):
    def __init__(self, in_channels, kernel_size=7, stride=1):
        super(ConvNext, self).__init__()
        padding = kernel_size//2
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels*4, kernel_size=1, stride=1, padding=0)
        self.gelu = torch.nn.GELU()
        self.conv3 = torch.nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn(z)
        z = self.conv2(z)
        z = self.gelu(z)
        z = self.conv3(z)
        z = z + x
        return z


class ConvNextModel(torch.nn.Module):

    def __init__(self, **kwargs):
        super(ConvNextModel, self).__init__()
        self.conv = SeparableConv(in_channels=3, out_channels=32, kernel_size=3, stride=4)
        self.block1 = self.convnext_block(num=3, in_channels=32)
        self.down_conv1 = SeparableConv(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.block2 = self.convnext_block(num=3, in_channels=64)
        self.down_conv2 = SeparableConv(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.block3 = self.convnext_block(num=9, in_channels=128)
        self.down_conv3 = SeparableConv(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.block4 = self.convnext_block(num=3, in_channels=256)
        self.channel_num = [64, 128, 256]

    def convnext_block(self, num, in_channels):
        layers = [ConvNext(in_channels) for i in range(num)]
        blocks = torch.nn.Sequential(*layers)
        return blocks
    
    def forward(self, x):
        outs = OrderedDict()
        x = self.conv(x)
        x = self.block1(x)
        x = self.down_conv1(x)
        x = self.block2(x)
        outs['feat0'] = x
        x = self.down_conv2(x)
        x = self.block3(x)
        outs['feat1'] = x
        x = self.down_conv3(x)
        x = self.block4(x)
        outs['feat2'] = x

        return outs


