from .centernet3d_loss import CenterNet3DLoss
from .centernet3d_depth_loss import CenterNet3DDepthLoss


def get_criterion(name):
    if name == "CenterNet3DLoss":
        return CenterNet3DLoss
    elif name=='CenterNet3DDepthLoss':
        return CenterNet3DDepthLoss
