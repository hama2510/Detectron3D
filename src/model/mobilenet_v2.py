import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import mobilenet_v2

class MobileNetv2(nn.Module):
    def __init__(self, device, pretrained=True):
        super().__init__()
        self.model = mobilenet_v2(pretrained=pretrained)
        self.model = self.model.to(device)
        self.channel_num = [32, 96, 320]
        
    def forward(self, x):
        outs = OrderedDict()
        outs['feat0'] = self.model.features[:7](x)    
        outs['feat1'] = self.model.features[:14](x)       
        outs['feat2'] = self.model.features[:-1](x)
        return outs