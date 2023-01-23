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
        # print(outs['p3'].shape, outs['p4'].shape, outs['p5'].shape, outs['p6'].shape, outs['p7'].shape, )
        return outs

class FusedFPN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(in_channel_list, out_channel)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x):
        outs = OrderedDict()
        x = self.fpn(x)
        x2_up = self.up(x['feat2'])
        x2_up = x2_up[:,:,:x['feat1'].shape[2],:]
#         print(x['feat1'].shape, x2_up.shape)
#         outs['p4'] = torch.sum((x['feat1'], x2_up), 1)
#         print(x2_up.shape, x['feat1'].shape)
        outs['p4'] = x['feat1']+x2_up
        return outs