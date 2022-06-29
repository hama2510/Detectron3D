import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import mobilenet_v2

class MobileNetv2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.module = mobilenet_v2(pretrained=pretrained).features,
        self.channel_num = [512, 1024, 2048]
        
    def forward(self, x):
        m = self.module
        outs = OrderedDict()
        outs['feat0'] = m[:7](x)    
        outs['feat1'] = m[:14](x)       
        outs['feat2'] = m[:-1](x)
        return outs