import torch
import torch.nn as nn
from collections import OrderedDict

class ResNet101(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=pretrained)
        self.channel_num = [512, 1024, 2048]
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        outs = OrderedDict()
        outs['feat0'] = x2    
        outs['feat1'] = x3        
        outs['feat2'] = x4
        return outs