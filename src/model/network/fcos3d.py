import torch
import torch.nn as nn
from collections import OrderedDict


class FCOS3D(nn.Module):
    def __init__(self, feature_extractor, fpn, num_cate, num_attr):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fpn = fpn(self.feature_extractor.channel_num, 256)
        self.cls_head = ClassificationHead(256, num_cate, num_attr, (3, 3), 1, 4)
        self.regress_head = RegressionHead(256, (3, 3), 1, 4)

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


class PredictionHead(nn.Module):
    def __init__(self, in_channel, kernel_size, padding, num_conv):
        super().__init__()
        self.num_conv = num_conv
        conv_list = []
        for i in range(self.num_conv):
            conv_list.append(
                nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding)
            )
            conv_list.append(nn.ReLU())
        self.convs = nn.Sequential(*conv_list)

    def forward(self, x):
        x = self.convs(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, in_channel, num_cate, num_attr, kernel_size, padding, num_conv):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.convs = PredictionHead(in_channel, kernel_size, padding, num_conv)
        self.conv_cate = nn.Conv2d(in_channel, num_cate, kernel_size, padding=padding)
        self.conv_attr = nn.Conv2d(in_channel, num_attr, kernel_size, padding=padding)

    def forward(self, x):
        x = self.convs(x)
        outs = OrderedDict()
        outs["category"] = self.conv_cate(x).sigmoid()
        outs["attribute"] = torch.nn.functional.softmax(self.conv_attr(x), dim=1)
        return outs


class RegressionHead(nn.Module):
    def __init__(self, in_channel, kernel_size, padding, num_conv):
        super().__init__()
        self.convs = PredictionHead(in_channel, kernel_size, padding, num_conv)
        self.relu = nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.conv_centerness = nn.Conv2d(in_channel, 1, kernel_size, padding=padding)
        self.conv_offset = nn.Conv2d(in_channel, 2, kernel_size, padding=padding)
        self.conv_depth = nn.Conv2d(in_channel, 1, kernel_size, padding=padding)
        self.conv_size = nn.Conv2d(in_channel, 3, kernel_size, padding=padding)
        self.conv_rotation = nn.Conv2d(in_channel, 1, kernel_size, padding=padding)
        self.conv_dir = nn.Conv2d(in_channel, 2, kernel_size, padding=padding)
        self.conv_velo = nn.Conv2d(in_channel, 2, kernel_size, padding=padding)

    def forward(self, x):
        x = self.convs(x)
        outs = OrderedDict()
        outs["centerness"] = self.conv_centerness(x).sigmoid()
        outs["offset"] = self.conv_offset(x)
        outs["depth"] = self.relu(self.conv_depth(x))
        outs["size"] = self.relu(self.conv_size(x))
        outs["rotation"] = self.tanh(self.conv_rotation(x))
        outs["dir"] = self.conv_dir(x).sigmoid()
        outs["velocity"] = self.conv_velo(x)
        return outs
