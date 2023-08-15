from .centernet3d_loss import CenterNet3DLoss


def get_criterion(name):
    if name == "CenterNet3DLoss":
        return CenterNet3DLoss
