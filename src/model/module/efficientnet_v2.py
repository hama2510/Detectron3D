import torch
import torch.nn as nn
from collections import OrderedDict

class EfficientNetV2S(nn.Module):
    def __init__(self, device, pretrained=True):
        super().__init__()
        self.model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=pretrained)
        self.model = self.model.to(device)
        self.channel_num = [64, 160, 256]
        
    def forward(self, x):
        modules = list(self.model.children())
        outs = OrderedDict()
        x = modules[0](x)
        for i, module in enumerate(modules[1]):
            x = module(x)
            if i==9:
                outs['feat0'] = x   
            elif i==24:
                outs['feat1'] = x
            elif i==39:
                outs['feat2'] = x
        return outs