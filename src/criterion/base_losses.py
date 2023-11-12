import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math


class BaseLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.esp = 1e-4

    def to_device(self, item):
        for key in item.keys():
            item[key] = item[key].to(self.device)
        return item

    def move_axis(self, pred, target):
        target = torch.moveaxis(target, -1, 1)
        return pred, target

    def gen_mask(self, target):
        target = torch.moveaxis(target, -1, 1)
        masked = target.sum(axis=1)
        return masked

    def focal_loss(self, pred, target, gamma=2):
        pred, target = self.move_axis(pred, target)
        pred = torch.clamp(pred, min=self.esp, max=1 - self.esp)

        pos_inds = target.eq(1).float()
        neg_inds = target.eq(0).float()

        loss = 0
#         neg_weights = torch.pow(1 - target, 4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_inds

        num_pos = pos_inds.float().sum()
        num_neg = neg_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
#             loss = loss - (pos_loss + neg_loss) / num_pos
            loss = loss - (pos_loss/num_pos + neg_loss/num_neg)
        return loss

    def cross_entropy_loss(self, pred, target, masked=True):
        pred, target = self.move_axis(pred, target)
        pred = torch.clamp(pred, min=self.esp, max=1 - self.esp)
        # print(pred.shape)
        pos_inds = target.eq(1).float()
        neg_inds = target.eq(0).float()

        loss = 0

        pos_loss = torch.log(pred) * pos_inds
        neg_loss = torch.log(1 - pred) * neg_inds

        # print(pred[[0],:,[0],[0]])
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            if masked:
                loss = loss - pos_loss / num_pos
            else:
                loss = loss - (pos_loss + neg_loss) / num_pos
#                 loss = loss - (pos_loss/num_pos + neg_loss/num_neg)
        return loss

    def smooth_l1_loss(self, pred, target, masked):
        pred, target = self.move_axis(pred, target)
        if masked is None:
            return nn.SmoothL1Loss()(pred, target)
        else:
            num_pos = masked.sum()
            if num_pos == 0:
                return 0
            else:
                return (
                    nn.SmoothL1Loss(reduction="none")(pred, target).mean(axis=1)
                    * masked
                ).sum() / num_pos

    def l1_loss(self, pred, target, masked):
        pred, target = self.move_axis(pred, target)
        if masked is None:
            return nn.L1Loss()(pred, target)
        else:
            num_pos = masked.sum()
            if num_pos == 0:
                return 0
            else:
                return (
                    nn.L1Loss(reduction="none")(pred, target).mean(axis=1) * masked
                ).sum() / num_pos

    def stride_to_feat_level(self, stride):
        return int(np.log2(stride))

    def forward(self, target, pred):
        raise NotImplementedError()
