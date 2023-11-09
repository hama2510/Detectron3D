import torch
import numpy as np
from .base_detector import BaseDetector
from model.network.centernet3D import CenterNet3D
from model.network.centernet3D_depth import CenterNetDepth3D


class CenterNet3Detector(BaseDetector):
    def create_head(self):
        if self.config.model.head_name == "CenterNet3D":
            head = CenterNet3D
        elif self.config.model.head_name=='CenterNetDepth3D':
            head = CenterNetDepth3D
        else:
            raise ValueError("Not supported detector name")
        return head

    def item_tensor_to_numpy(self, key, item):
        if key == "category":
            item = torch.clamp(item, min=1e-4, max=None)
        item = item.detach().cpu().numpy()
        item = np.moveaxis(item, 0, -1)
        return item

    def get_head_keys(self):
        return [
            "category",
            "attribute",
            "offset",
            "depth",
            "size",
            "rotation",
            "velocity",
        ]
