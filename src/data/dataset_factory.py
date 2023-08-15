from .nuscene_dataset_fcos3d import NusceneDatasetFCOS3D
from .nuscene_dataset_centernet import NusceneDatasetCenterNet


def get_dataset(name):
    if name == "NusceneDatasetFCOS3D":
        return NusceneDatasetFCOS3D
    elif name == "NusceneDatasetCenterNet":
        return NusceneDatasetCenterNet
