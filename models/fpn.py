import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class FPN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(in_channel_list, out_channel)
        self.conv = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, 
                               kernel_size=(3,3), stride=2, padding=1)
    
    def forward(self, x):
        outs = OrderedDict()
        x = self.fpn(x)
        outs['p3'] = x['feat0']
        outs['p4'] = x['feat1']        
        outs['p5'] = x['feat2']        
        outs['p6'] = self.conv(outs['p5'])
        outs['p7'] = self.conv(outs['p6'])   
        return outs