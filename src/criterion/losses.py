import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math
from base_losses import BaseLoss
# class Criterion(nn.Module):
#     def __init__(self, device, alpha=0.2, gamma=2):
#         super().__init__()
#         self.device = device
#         self.alpha = alpha
#         self.gamma = gamma
#         self.esp = 1e-4
        
#     def to_device(self, item):
#         for key in item.keys():
#             item[key] = item[key].to(self.device)
#         return item

#     def flattern(self, pred, target):
#         pred = pred.view(pred.shape[0], pred.shape[1], pred.shape[2]*pred.shape[3])
#         target = torch.moveaxis(target, -1, 1)
#         target = target.view(target.shape[0], target.shape[1], target.shape[2]*target.shape[3])
#         return pred, target
    
#     def gen_mask(self, target):
#         target = torch.moveaxis(target, -1, 1)
#         target = target.view(target.shape[0], target.shape[1], target.shape[2]*target.shape[3])
#         masked = target.sum(axis=1)
# #         return masked.view(masked.shape[0], 1, masked.shape[1])
#         return masked
        
#     def focal_loss(self, pred, target):
#         pred, target = self.flattern(pred, target)
#         pred = torch.clamp(pred, min=self.esp, max=1-self.esp)

#         pos_inds = target.eq(1)
#         neg_inds = target.lt(1)
        
#         pos_pred = pred[pos_inds]
#         neg_pred = pred[neg_inds]

#         pos_loss = self.alpha * torch.log(pos_pred) * torch.pow(1 - pos_pred, self.gamma)
#         neg_loss = self.alpha * torch.log(1 - neg_pred) * torch.pow(neg_pred, self.gamma)
#         # neg_weights = torch.pow(1 - target[neg_inds], 4)
#         # neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, self.gamma) * neg_weights

#         num_pos  = pos_inds.float().sum()
#         pos_loss = pos_loss.sum()
#         neg_loss = neg_loss.sum()

#         if num_pos == 0:
#             loss = 0 - neg_loss
#         else:
#             loss = 0 - (pos_loss + neg_loss) / num_pos
#         return loss

#     def smooth_l1_loss(self, pred, target, masked):
#         pred, target = self.flattern(pred, target)
#         if masked is None:
#             return nn.SmoothL1Loss()(pred, target)
#         else:
#             num_pos = masked.sum()
#             if num_pos==0:
#                 return 0
#             else:
#                 return (nn.SmoothL1Loss(reduction='none')(pred, target).mean(axis=1)*masked).sum()/num_pos
            
#     def l1_loss(self, pred, target, masked):
#         pred, target = self.flattern(pred, target)
#         if masked is None:
#             return nn.L1Loss()(pred, target)
#         else:
#             num_pos = masked.sum()
#             if num_pos==0:
#                 return 0
#             else:
#                 return (nn.L1Loss(reduction='none')(pred, target).mean(axis=1)*masked).sum()/num_pos

#     def bce_loss(self, pred, target, masked):
#         pred, target = self.flattern(pred, target)
#         pred = torch.clamp(pred, min=self.esp, max=1-self.esp)
#         if masked is None:
#             return nn.BCELoss()(pred, target)
#         else:
#             num_pos = masked.sum()
#             if num_pos==0:
#                 return 0
#             else:
#                 return (nn.BCELoss(reduction='none')(pred, target)*masked).sum()/num_pos

# #     def kl_loss(self, pred, target):
# #         pred, target = self.flattern(pred, target)
# #         pred = torch.clamp(pred, min=1e-4, max=1-1e-4)
# #         kl_loss = target*torch.log(target/pred)
# #         # pos_inds = target.eq(1)
# #         # loss = nn.BCELoss(reduction='none')(pred, target).mean(axis=1)*pos_index.float().sum(axis=1)
# #         # loss = loss.sum()/pos_inds.float().sum()
# #         return loss

# #     def cross_entropy_loss(self, pred, target, masked):
# #         pred, target = self.flattern(pred, target)
# # #         pred = torch.clamp(pred, min=self.esp, max=1-self.esp)
# #         if masked is None:
# #             return nn.CrossEntropyLoss()(pred, target)
# #         else:
# #             num_pos = masked.sum()
# #             if num_pos==0:
# #                 return 0
# #             else:
# #                 return (nn.CrossEntropyLoss(reduction='none')(pred, target)*masked).sum()/num_pos
            
#     def cross_entropy_loss(self, pred, target, masked):
#         pred, target = self.flattern(pred, target)
#         pred = torch.clamp(pred, min=self.esp, max=1-self.esp)
        
#         pos_inds = target.eq(1)
#         neg_inds = target.lt(1)
        
#         pos_pred = pred[pos_inds]
#         neg_pred = pred[neg_inds]

#         pos_loss = torch.log(pos_pred)
#         neg_loss = torch.log(1 - neg_pred)

#         num_pos  = pos_inds.float().sum()
#         pos_loss = pos_loss.sum()
#         neg_loss = neg_loss.sum()

#         if num_pos == 0:
#             loss = 0 - neg_loss
#         else:
#             loss = 0 - (pos_loss + neg_loss) / num_pos
#         return loss

#     def stride_to_feat_level(self, stride):
#         return int(np.log2(stride))

#     def loss(self, pred, target, stride):
#         if not 'p{}'.format(self.stride_to_feat_level(stride)) in pred.keys():
#             return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#         pred = pred['p{}'.format(self.stride_to_feat_level(stride))]
#         target = target['{}'.format(stride)]
#         target = self.to_device(target)
        
#         masked = self.gen_mask(target['category'])
# #         print(masked.sum())

#         category_loss = self.focal_loss(pred['category'], target['category'])
#         attribute_loss = self.cross_entropy_loss(pred['attribute'], target['attribute'], masked)
#         centerness_loss = self.cross_entropy_loss(pred['centerness'], target['centerness'], masked)
#         offset_loss = self.smooth_l1_loss(pred['offset'], target['offset'], masked)
#         depth_loss = self.smooth_l1_loss(torch.exp(pred['depth']), target['depth'], masked)
#         size_loss = self.smooth_l1_loss(pred['size'], target['size'], masked)
#         rotation_loss = self.smooth_l1_loss(pred['rotation'], target['rotation'], masked)
#         dir_loss = self.cross_entropy_loss(pred['dir'], target['dir'], masked)
#         velocity_loss = self.smooth_l1_loss(pred['velocity'], target['velocity'], masked)
# #         print(category_loss, attribute_loss, offset_loss, depth_loss, size_loss, rotation_loss, velocity_loss, dir_loss, centerness_loss)
#         return category_loss, attribute_loss, offset_loss, depth_loss, size_loss, rotation_loss, velocity_loss, dir_loss, centerness_loss

        
#     def forward(self, target, pred):
#         total_loss = 0
#         loss_log = {}
#         for stride in target.keys():
#             category_loss, attribute_loss, offset_loss, depth_loss, size_loss, rotation_loss, velocity_loss, dir_loss, centerness_loss = self.loss(pred, target, int(stride))
#             loss = category_loss+attribute_loss+(offset_loss+0.2*depth_loss+size_loss+2*rotation_loss+0.05*velocity_loss)+dir_loss+centerness_loss
#             total_loss+=loss
            
#             loss_log[stride] = {
#                 'category_loss': category_loss.cpu().detach().numpy() if torch.is_tensor(category_loss) else category_loss,
#                 'attribute_loss': attribute_loss.cpu().detach().numpy() if torch.is_tensor(attribute_loss) else attribute_loss,
#                 'offset_loss': offset_loss.cpu().detach().numpy() if torch.is_tensor(offset_loss) else offset_loss,
#                 'depth_loss': depth_loss.cpu().detach().numpy() if torch.is_tensor(depth_loss) else depth_loss,
#                 'size_loss': size_loss.cpu().detach().numpy() if torch.is_tensor(size_loss) else size_loss,
#                 'rotation_loss': rotation_loss.cpu().detach().numpy() if torch.is_tensor(rotation_loss) else rotation_loss,
#                 'dir_loss': dir_loss.cpu().detach().numpy() if torch.is_tensor(dir_loss) else dir_loss,
#                 'centerness_loss': centerness_loss.cpu().detach().numpy() if torch.is_tensor(centerness_loss) else centerness_loss,
#             }
#         return total_loss, loss_log

class FCOS3DLoss(BaseLoss):
    def loss(self, pred, target, stride):
        if not "p{}".format(self.stride_to_feat_level(stride)) in pred.keys():
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        pred = pred["p{}".format(self.stride_to_feat_level(stride))]
        target = target["{}".format(stride)]
        target = self.to_device(target)

        masked = self.gen_mask(target["category"])

        category_loss = self.focal_loss(pred["category"], target["category"])
        attribute_loss = self.cross_entropy_loss(
            pred["attribute"], target["attribute"]
        )

        offset_loss = self.smooth_l1_loss(pred["offset"], target["offset"], masked)
        depth_loss = self.smooth_l1_loss(
            torch.exp(pred["depth"]), target["depth"], masked
        )
        size_loss = self.smooth_l1_loss(pred["size"], target["size"], masked)
        rotation_loss = self.smooth_l1_loss(
            pred["rotation"], target["rotation"], masked
        )
        velocity_loss = self.smooth_l1_loss(
            pred["velocity"], target["velocity"], masked
        )
        centerness_loss = self.cross_entropy_loss(pred['centerness'], target['centerness'])
        return (
            category_loss,
            attribute_loss,
            offset_loss,
            depth_loss,
            size_loss,
            rotation_loss,
            velocity_loss,
            centerness_loss
        )

    def forward(self, target, pred):
        total_loss = 0
        loss_log = {}
        for stride in target.keys():
            (
                category_loss,
                attribute_loss,
                offset_loss,
                depth_loss,
                size_loss,
                rotation_loss,
                velocity_loss,
                centerness_loss
            ) = self.loss(pred, target, int(stride))
            loss = (
                category_loss
                + attribute_loss
                + (
                    offset_loss
                    + depth_loss
                    + size_loss
                    + rotation_loss
                    + 0.05 * velocity_loss
                    + centerness_loss
                )
            )
            total_loss += loss

            loss_log[stride] = {
                "category_loss": category_loss.cpu().detach().numpy()
                if torch.is_tensor(category_loss)
                else category_loss,
                "attribute_loss": attribute_loss.cpu().detach().numpy()
                if torch.is_tensor(attribute_loss)
                else attribute_loss,
                "offset_loss": offset_loss.cpu().detach().numpy()
                if torch.is_tensor(offset_loss)
                else offset_loss,
                "depth_loss": depth_loss.cpu().detach().numpy()
                if torch.is_tensor(depth_loss)
                else depth_loss,
                "size_loss": size_loss.cpu().detach().numpy()
                if torch.is_tensor(size_loss)
                else size_loss,
                "rotation_loss": rotation_loss.cpu().detach().numpy()
                if torch.is_tensor(rotation_loss)
                else rotation_loss,
                "dir_loss": 0,
                "centerness_loss": centerness_loss.cpu().detach().numpy(),
            }
        return total_loss, loss_log
