import torch
import torchvision.ops
from torch import nn
from dcn import DeformableConv2d
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, mask, num_block):
        super().__init__()
        
        self.num_block = num_block
        self.bn_mid = nn.BatchNorm2d(mid_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv1_stride = DeformableConv2d(in_channels, mid_channels, kernel_size=1, stride=stride, 
                                        padding=0, mask=mask)
        self.conv1 = DeformableConv2d(out_channels, mid_channels, kernel_size=1, stride=1, 
                                padding=0, mask=mask)
        self.conv2 = DeformableConv2d(mid_channels, mid_channels, kernel_size=3, stride=1, 
                                        padding=1, mask=mask)
        self.conv3 = DeformableConv2d(mid_channels, out_channels, kernel_size=1, stride=1, 
                                        padding=0, mask=mask)
        self.conv_projection = DeformableConv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                                        padding=0, mask=mask)
        self.conv_projection_stride = DeformableConv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                                        padding=0, mask=mask)
        
    def forward(self, x):
        for i in range(self.num_block):
            identity = x
            x = self.relu(self.bn_mid(self.conv1_stride(x))) if i==0 else self.relu(self.bn_mid(self.conv1(x)))
            x = self.relu(self.bn_mid(self.conv2(x)))
            x = self.conv3(x)
            if x.size()[1]!=identity.size()[1]:
                identity = self.conv_projection_stride(identity) if i==0 else self.conv_projection(identity)
            x = torch.add(x, identity)
            x = self.bn_out(x)
            x = self.relu(x)
        return x
        
class ResNet101DCN(nn.Module):
    def __init__(self, mask=False):
        super().__init__()
        
        self.channel_num = [512, 1024, 2048]
        self.mask = mask
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = DeformableConv2d(3, 64, kernel_size=7, stride=2, padding=3, mask=self.mask)
        self.conv2 = ResidualBlock(64, 64, 256, mask=mask, stride=1, num_block=3)
        self.conv3 = ResidualBlock(256, 128, 512, mask=mask, stride=2, num_block=4)
        self.conv4 = ResidualBlock(512, 256, 1024, mask=mask, stride=2, num_block=6)
        self.conv5 = ResidualBlock(1024, 512, 2048, mask=mask, stride=2, num_block=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)
        x4 = self.conv5(x3)
        
        outs = OrderedDict()
        outs['feat0'] = x2   
        outs['feat1'] = x3        
        outs['feat2'] = x4
        return outs