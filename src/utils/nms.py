from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from .camera import box_3d_to_2d, xywh_to_xyxy
from pyquaternion import Quaternion
# import open3d.ml.tf as ml3d
from mmcv.ops.nms import nms_rotated
import numpy as np
import torch

def rotated_nms(boxes, calibration_matrix,  nms_thres=0.3):
    boxes_2d = []
    scores = []
    for box in boxes:
        bb = Box(box['translation'], box['size'], Quaternion(box['rotation']))
        bb_2d = box_3d_to_2d(bb, calibration_matrix)
        boxes_2d.append([bb_2d[0][0], bb_2d[0][1], bb_2d[1], bb_2d[2], box['rotation_angle']])
        scores.append(box['detection_score'])
    boxes_2d = np.asarray(boxes_2d)
    scores = np.asarray(scores)
#     keep_indices = ml3d.ops.nms(boxes_2d, scores, nms_thres)
    _, keep_indices = nms_rotated(torch.tensor(boxes_2d).cuda(), torch.tensor(scores).cuda(), iou_threshold=nms_thres)
    if not keep_indices is None:
        return keep_indices.detach().cpu().numpy()
    else:
        return []