from .nuscene_dataset_fcos3d import NusceneDatasetFCOS3D
from .nuscene_dataset_centernet import NusceneDatasetCenterNet
from .nuscene_dataset_centernet_depth import NusceneDatasetCenterNetDepth

def get_dataset(name):
    if name == "NusceneDatasetFCOS3D":
        return NusceneDatasetFCOS3D
    elif name == "NusceneDatasetCenterNet":
        return NusceneDatasetCenterNet
    elif name == 'NusceneDatasetCenterNetDepth':
        return NusceneDatasetCenterNetDepth
    else:
        raise ValueError('Not supported dataset')
