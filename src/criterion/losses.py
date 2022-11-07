import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math

class Criterion(nn.Module):
    def __init__(self, device, alpha=0.2, gamma=2):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        
    def focal_loss(self, pred, target, esp=1e-8):
        pred, target = self.flattern(pred, target)
        pred = torch.clamp(torch.nn.functional.softmax(pred,dim=2), min=1e-4, max=1-1e-4)

        pos_index = torch.eq(target, 1)
        pt = 1 - pred
        pt[pos_index] = pred[pos_index]
        pt = torch.clamp(pt, esp, 1-esp)
        loss = -1*self.alpha*(torch.pow((1-pt), self.gamma))*torch.log(pt)
        loss = loss.mean()
        return loss

    def to_device(self, item):
        for key in item.keys():
            item[key] = item[key].to(self.device)
        return item

    def flattern(self, pred, target):
        shape = pred.shape
        pred = torch.reshape(pred, (shape[0], shape[1], shape[2]*shape[3])).view(shape[0], shape[2]*shape[3], shape[1])
        shape = target.shape
        target = torch.reshape(target, (shape[0], shape[1]*shape[2], shape[3]))
        return pred, target

    def smooth_l1_loss(self, pred, target):
        pred, target = self.flattern(pred, target)
        return nn.SmoothL1Loss()(pred, target)

    def bce_loss(self, pred, target):
        pred, target = self.flattern(pred, target)
        return nn.BCELoss()(pred, target)

    def cross_entropy_loss(self, pred, target):
        pred, target = self.flattern(pred, target)
        pred = torch.clamp(torch.nn.functional.softmax(pred,dim=2), min=1e-4, max=1-1e-4)
        return nn.CrossEntropyLoss()(pred, target)

    def stride_to_feat_level(self, stride):
        return int(np.log2(stride))

    def loss(self, pred, target, stride):
        pred = pred['p{}'.format(self.stride_to_feat_level(stride))]
        target = target['{}'.format(stride)]
        target = self.to_device(target)

        category_loss = self.focal_loss(pred['category'], target['category'])
        attribute_loss = self.cross_entropy_loss(pred['attribute'], target['attribute'])
        centerness_loss = self.bce_loss(pred['centerness'], target['centerness'])
        offset_loss = self.smooth_l1_loss(pred['offset'], target['offset'])
        depth_loss = self.smooth_l1_loss(torch.exp(pred['depth']), target['depth'])
        size_loss = self.smooth_l1_loss(pred['size'], target['size'])
        rotation_loss = self.smooth_l1_loss(pred['rotation'], target['rotation'])
        dir_loss = self.cross_entropy_loss(pred['dir'], target['dir'])
        velocity_loss = self.smooth_l1_loss(pred['velo'], target['velocity'])
#         return category_loss+attribute_loss+(offset_loss+0.2*depth_loss+size_loss+rotation_loss+0.05*velocity_loss)+dir_loss+centerness_loss
        return category_loss, attribute_loss, offset_loss, depth_loss, size_loss, rotation_loss, velocity_loss, dir_loss, centerness_loss
    
    def forward(self, target, pred):
        total_loss = 0
        loss_log = {}
        for stride in target.keys():
            category_loss, attribute_loss, offset_loss, depth_loss, size_loss, rotation_loss, velocity_loss, dir_loss, centerness_loss = self.loss(pred, target, int(stride))
            loss = category_loss+attribute_loss+(offset_loss+0.2*depth_loss+size_loss+rotation_loss+0.05*velocity_loss)+dir_loss+centerness_loss
            total_loss+=loss
            loss_log[stride] = {
                'category_loss': category_loss.cpu().detach().numpy(),
                'attribute_loss': attribute_loss.cpu().detach().numpy(),
                'offset_loss': offset_loss.cpu().detach().numpy(),
                'depth_loss': depth_loss.cpu().detach().numpy(),
                'size_loss': size_loss.cpu().detach().numpy(),
                'rotation_loss': rotation_loss.cpu().detach().numpy(),
                'dir_loss': dir_loss.cpu().detach().numpy(),
                'centerness_loss': centerness_loss.cpu().detach().numpy(),
            }
        return total_loss, loss_log