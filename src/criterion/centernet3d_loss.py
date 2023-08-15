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
            torch.exp(pred["depth"]), target["depth"], masked
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
                    + 0.2 * depth_loss
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
