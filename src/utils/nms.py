from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from .camera import box_3d_to_2d, xywh_to_xyxy
import open3d.ml.tf as ml3d
import numpy as np

def rotated_nms(boxes, calibration_matrix,  nms_thres=0.3):
    boxes_2d = []
    scores = []
    for box in boxes:
        bb = Box(box['translation'], box['size'], box['rotation'])
        bb_2d = xywh_to_xyxy(box_3d_to_2d(bb, calibration_matrix))
        bb_2d.append(box['rotation_angle'])
        boxes_2d.append(bb_2d)
        scores.append(box['detection_score'])
    boxes_2d = np.asarray(boxes_2d)
    scores = np.asarray(scores)
    keep_indices = ml3d.ops.nms(boxes_2d, scores, nms_thres)
    return keep_indices