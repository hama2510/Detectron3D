import torch
import torch.nn as nn
import numpy as np
from model.module.fpn import FPN, FusedFPN, FusedFPNP3
from collections import OrderedDict
# from .fcos3d import PredictionHead

class CenterNet3DShallow(nn.Module):
    def __init__(self, feature_extractor, num_cate, num_attr):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fpn = FusedFPN(self.feature_extractor.channel_num, 256)
        self.cls_head = ClassificationHead(256, num_cate, num_attr, (3,3), 1)
        self.regress_head = RegressionHead(256, (3,3), 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.fpn(features)
        outs = OrderedDict()
        for key in features.keys():
            outs[key] = OrderedDict()
            x_cls = self.cls_head(features[key])
            for k in x_cls.keys():
                outs[key][k] = x_cls[k]
            x_regress = self.regress_head(features[key])
            for k in x_regress.keys():
                outs[key][k] = x_regress[k]
        return outs
    
# class PredictionHead(nn.Module):
#     def __init__(self, in_channel, kernel_size, padding, num_conv):
#         super().__init__()
#         self.num_conv = num_conv
#         conv_list = []
#         for i in range(self.num_conv):
#             conv_list.append(nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding))
#             conv_list.append(nn.ReLU())
#         self.convs = nn.Sequential(*conv_list)

#     def forward(self, x):
#         x = self.convs(x)
#         return x

class ClassificationHead(nn.Module):
    def __init__(self, in_channel, num_cate, num_attr, kernel_size, padding):
        super().__init__()
        self.relu = nn.ReLU()
        # self.convs = PredictionHead(in_channel, kernel_size, padding, num_conv)
        self.conv_cate = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, num_cate, kernel_size, padding=padding))
        self.conv_attr = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, num_attr, 1, 1, 0))

    def forward(self, x):
        # x = self.convs(x)
        outs = OrderedDict()
        outs['category'] = self.conv_cate(x).sigmoid()
        outs['attribute'] = torch.nn.functional.softmax(self.conv_attr(x), dim=1)
        return outs

class RegressionHead(nn.Module):
    def __init__(self, in_channel, kernel_size, padding):
        super().__init__()
        # self.convs = PredictionHead(in_channel, kernel_size, padding, num_conv)
        self.relu = nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.conv_offset = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, 2, kernel_size, padding=padding))
        self.conv_depth = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, 1, kernel_size, padding=padding))
        self.conv_size = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, 3, kernel_size, padding=padding))
        self.conv_rotation = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, 1, kernel_size, padding=padding))
        self.conv_velo = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, 2, kernel_size, padding=padding)
        )

    def forward(self, x):
        # x = self.convs(x)
        outs = OrderedDict()
        outs['offset'] = self.conv_offset(x)
        outs['depth'] = self.relu(self.conv_depth(x))
        outs['size'] = self.relu(self.conv_size(x))
        outs['rotation'] = self.tanh(self.conv_rotation(x))
        outs['velocity'] = self.conv_velo(x)
        return outs
