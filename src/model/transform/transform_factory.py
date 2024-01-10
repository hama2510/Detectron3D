from .fcos_transform import FCOSTransformer
from .centernet_transform import CenterNetTransformer
from .centernet_transform_mask_filter import CenterNetTransformerMaskFilter
from .centernet_transform_depth import CenterNetTransformerDepth

def get_transform(name):
    if name == "FCOSTransformer":
        return FCOSTransformer
    elif name == "CenterNetTransformer":
        return CenterNetTransformer
    elif name == 'CenterNetTransformerMaskFilter':
        return CenterNetTransformerMaskFilter
    elif name=='CenterNetTransformerDepth':
        return CenterNetTransformerDepth
    else:
        raise ValueError('Not supported transformer')