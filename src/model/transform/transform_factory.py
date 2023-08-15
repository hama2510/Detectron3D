from .fcos_transform import FCOSTransformer
from .centernet_transform import CenterNetTransformer


def get_transform(name):
    if name == "FCOSTransformer":
        return FCOSTransformer
    elif name == "CenterNetTransformer":
        return CenterNetTransformer
