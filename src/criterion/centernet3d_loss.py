import torch
from .base_losses import BaseLoss


class CenterNet3DLoss(BaseLoss):
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
            torch.exp(pred["depth"]), target["depth"]
        )
        size_loss = self.smooth_l1_loss(pred["size"], target["size"], masked)
        rotation_loss = self.smooth_l1_loss(
            pred["rotation"], target["rotation"], masked
        )
        velocity_loss = self.smooth_l1_loss(
            pred["velocity"], target["velocity"], masked
        )
        return (
            category_loss,
            attribute_loss,
            offset_loss,
            depth_loss,
            size_loss,
            rotation_loss,
            velocity_loss,
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
                "centerness_loss": 0,
            }
        return total_loss, loss_log

class CenterNet3DLossWithDcMask(CenterNet3DLoss):
    def gen_mask(self, target):
        target = torch.moveaxis(target, -1, 1)
        masked = target.sum(axis=1)
        masked = torch.clamp(masked, min=0, max=1)
        return masked

    def focal_loss(self, pred, target, mask, gamma=2, ):
        pred, target = self.move_axis(pred, target)
        pred = torch.clamp(pred, min=self.esp, max=1 - self.esp)

        pred = pred*mask
        target = target*mask

        pos_inds = target.gt(0).float()
        neg_inds = target.eq(0).float()

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def cross_entropy_loss(self, pred, target):
        pred, target = self.move_axis(pred, target)
        pred = torch.clamp(pred, min=self.esp, max=1 - self.esp)

        pos_inds = target.gt(0).float()
        neg_inds = target.eq(0).float()

        loss = 0

        pos_loss = torch.log(pred) * pos_inds
        neg_loss = torch.log(1 - pred) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss