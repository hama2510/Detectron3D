import cv2 as cv
import os
from pyquaternion import Quaternion
# from scipy.spatial.transform import Rotation as quaternion_transformer
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box

def gen_calibration_matrix(intrinsic, sensor_R, sensor_t, ego_R, ego_t, calibrated=True):
    intrinsic, sensor_R, sensor_t, ego_R, ego_t = np.array(intrinsic), np.array(sensor_R), np.array(sensor_t), np.array(ego_R), np.array(ego_t)
    
    intrinsic = np.vstack([intrinsic, [0,0,0]])
    intrinsic = np.hstack([intrinsic, [[0],[0],[0],[1]]])
    
    if not calibrated:
        sensor_extrinsic = np.concatenate([
                np.float32(sensor_R),
                np.float32(sensor_t)[:, None],
            ], axis=-1)
        sensor_extrinsic = np.vstack([sensor_extrinsic, np.array([0, 0, 0, 1])])
        ego_extrinsic = np.concatenate([
            np.float32(ego_R),
            np.float32(ego_t)[:, None],
        ], axis=-1)
        ego_extrinsic = np.vstack([ego_extrinsic, np.array([0, 0, 0, 1])])
        return intrinsic, sensor_extrinsic, ego_extrinsic
    else:
        return intrinsic

def coord_3d_to_2d(coord_3d, calibration_matrix, calibrated=True):
    intrinsic, sensor_R, sensor_t, ego_R, ego_t = calibration_matrix['camera_intrinsic'], calibration_matrix['sensor_R'], calibration_matrix['sensor_t'], calibration_matrix['ego_R'], calibration_matrix['ego_t']
    coord_3d = np.array(coord_3d)
    coord_3d = np.append(coord_3d, 1)
    
    # no need extrinsic matrix because the coordinate has been rotated and translated
    if calibrated:
        intrinsic = gen_calibration_matrix(intrinsic, sensor_R, sensor_t, ego_R, ego_t, calibrated=calibrated)
        coord_2d = intrinsic @ coord_3d
    else:
        intrinsic, sensor_extrinsic, ego_extrinsic = gen_calibration_matrix(intrinsic, sensor_R, sensor_t, ego_R, ego_t, calibrated=calibrated)
        coord_2d = intrinsic @ sensor_extrinsic @ ego_extrinsic @ coord_3d
    coord_2d = (coord_2d / coord_2d[[-2]])[:2].T.astype(int)
    return coord_2d

def box_3d_to_2d(box, calibration_matrix, calibrated=True):
    c = coord_3d_to_2d(box.center, calibration_matrix)
    corners = box.corners()
    corners_2d = np.asarray([coord_3d_to_2d(corners[:,i], calibration_matrix, calibrated=calibrated) for i in range(corners.shape[1])])
    x_min = np.min(corners_2d[:,0])
    x_max = np.max(corners_2d[:,0])
    y_min = np.min(corners_2d[:,1])
    y_max = np.max(corners_2d[:,1])
    return [c, x_max-x_min, y_max-y_min]

def xywh_to_xyxy(box):
    c, w, h = box
    return [[c[0]-box[1], c[1]-box[2]], [c[0]+box[1], c[0]+box[2]]]

def is_inside(point, box):
    x,y = point
    (x_min, y_min), (x_max, y_max) = xywh_to_xyxy(box)
    if x<=x_max and x>=x_min and y<=y_max and y>=y_min:
        return True

def distance_to_center(point, box):
    return np.sum((point[0]-box[0][0])**2+(point[1]-box[0][1])**2)

def cal_area(box):
    return box[1]*box[2]

def coord_2d_to_3d(coord_2d, depth, calibration_matrix, calibrated=True):
    intrinsic, sensor_R, sensor_t, ego_R, ego_t = calibration_matrix['camera_intrinsic'], calibration_matrix['sensor_R'], calibration_matrix['sensor_t'], calibration_matrix['ego_R'], calibration_matrix['ego_t']
    intrinsic = gen_calibration_matrix(intrinsic, sensor_R, sensor_t, ego_R, ego_t, calibrated=calibrated)
    coord_2d = np.asarray(coord_2d)
    coord_2d = np.append(coord_2d, [1, 1/depth])
    intrinsic_inverse = np.linalg.inv(intrinsic)
    coord_3d = depth * (intrinsic_inverse @ coord_2d)
    coord_3d = coord_3d[:3].T
    return coord_3d

def sensor_coord_to_real_coord(coord, size, rotation, calibration_matrix):
    sensor_R_quaternion, sensor_t, ego_R_quaternion, ego_t = calibration_matrix['sensor_R_quaternion'], calibration_matrix['sensor_t'], calibration_matrix['ego_R_quaternion'], calibration_matrix['ego_t']
    box = Box(coord, size, rotation)
    box.rotate(Quaternion(sensor_R_quaternion))
    box.translate(np.array(sensor_t))
    box.rotate(Quaternion(ego_R_quaternion))
    box.translate(np.array(ego_t))
    return box.center
