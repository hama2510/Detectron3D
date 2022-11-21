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
        # pred = torch.clamp(torch.nn.functional.softmax(pred,dim=2), min=1e-4, max=1-1e-4)
        # pred = torch.clamp(pred, min=1e-4, max=1-1e-4)
        pos_inds = target.eq(1)
        neg_inds = target.lt(1)

        # neg_weights = torch.pow(1 - target[neg_inds], 4)

        loss = 0
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, self.gamma)
        # neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, self.gamma) * neg_weights
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, self.gamma)

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def to_device(self, item):
        for key in item.keys():
            item[key] = item[key].to(self.device)
        return item

    def flattern(self, pred, target):
        shape = pred.shape
        pred = torch.reshape(pred.view(shape[0], shape[2], shape[3], shape[1]), (shape[0]*shape[1]*shape[2], shape[3]))
        shape = target.shape
        target = torch.reshape(target, (shape[0]*shape[1]*shape[2], shape[3]))
        return pred, target

    def smooth_l1_loss(self, pred, target):
        pred, target = self.flattern(pred, target)
        return nn.SmoothL1Loss()(pred, target)

    def bce_loss(self, pred, target):
        pred, target = self.flattern(pred, target)
        loss = nn.BCELoss()(pred, target)
        # pos_inds = target.eq(1)
        # loss = nn.BCELoss(reduction='none')(pred, target)*pos_index.float().sum(axis=1)
        # loss = loss.sum()/pos_inds.float().sum()
        return loss

    def kl_loss(self, pred, target):
        pred, target = self.flattern(pred, target)
        pred = torch.clamp(pred, min=1e-4, max=1-1e-4)
        kl_loss = target*torch.log(target/pred)
        # pos_inds = target.eq(1)
        # loss = nn.BCELoss(reduction='none')(pred, target).mean(axis=1)*pos_index.float().sum(axis=1)
        # loss = loss.sum()/pos_inds.float().sum()
        return loss

    def cross_entropy_loss(self, pred, target):
        pred, target = self.flattern(pred, target)
        loss = nn.CrossEntropyLoss()(pred, target)
        # pred = torch.clamp(torch.nn.functional.softmax(pred,dim=2), min=1e-4, max=1-1e-4)
        # pos_inds = target.eq(1)
        # loss = nn.CrossEntropyLoss(reduction='none')(pred, target).mean(axis=1)*pos_inds.float().sum(axis=1)
        # loss = loss.sum()/.float().sum()
        return loss

    def stride_to_feat_level(self, stride):
        return int(np.log2(stride))

    def loss(self, pred, target, stride):
        pred = pred['p{}'.format(self.stride_to_feat_level(stride))]
        target = target['{}'.format(stride)]
        target = self.to_device(target)

        category_loss = self.focal_loss(pred['category'], target['category'])
        attribute_loss = self.cross_entropy_loss(pred['attribute'], target['attribute'])
        centerness_loss = self.bce_loss(pred['centerness'], target['centerness'])
        # centerness_loss = self.kl_loss(pred['centerness'], target['centerness'])
        offset_loss = self.smooth_l1_loss(pred['offset'], target['offset'])
        depth_loss = self.smooth_l1_loss(torch.exp(pred['depth']), target['depth'])
        size_loss = self.smooth_l1_loss(pred['size'], target['size'])
        rotation_loss = self.smooth_l1_loss(pred['rotation'], target['rotation'])
        dir_loss = self.cross_entropy_loss(pred['dir'], target['dir'])
        velocity_loss = self.smooth_l1_loss(pred['velocity'], target['velocity'])
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