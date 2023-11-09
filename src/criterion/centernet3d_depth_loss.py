import torch
from .centernet3d_loss import CenterNet3DLoss


class CenterNet3DDepthLoss(CenterNet3DLoss):
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
        depth_loss = self.cross_entropy_loss(
            pred["depth"], target["depth"], masked
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

