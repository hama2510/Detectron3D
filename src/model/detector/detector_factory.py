from .fcos3d_detector import FCOSDetector
from .centernet3d_detector import CenterNet3Detector


def get_detector(name):
    if name == "FCOSDetector":
        return FCOSDetector
    elif name == "CenterNet3Detector":
        return CenterNet3Detector
