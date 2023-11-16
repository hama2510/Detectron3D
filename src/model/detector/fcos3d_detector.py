import torch
import torch.nn as nn
import numpy as np
from model.network.fcos3d import FCOS3D
from .base_detector import BaseDetector


class FCOSDetector(BaseDetector):
    def create_head(self):
        if self.config.model.head_name == "FCOS3D":
            head = FCOS3D
        else:
            raise ValueError(f'Not implemeted {head}')
        return head

    def item_tensor_to_numpy(self, key, item):
        if key == "category":
            item = torch.clamp(item, min=1e-4, max=1 - 1e-4)
        elif key == "attribute" or key == "dir":
            item = nn.functional.softmax(item, dim=1)
        item = item.detach().cpu().numpy()
        item = np.moveaxis(item, 0, -1)
        return item

    def get_head_keys(self):
        return [
            "category",
            "attribute",
            "centerness",
            "offset",
            "depth",
            "size",
            "rotation",
            "dir",
            "velocity",
        ]
