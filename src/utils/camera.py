import cv2 as cv
import os
from scipy.spatial.transform import Rotation as quaternion_transformer
import numpy as np
from nuscenes.utils.geometry_utils import view_points

def quaternion_to_rotation_matrix(Q):
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
    # r00 = 2 * (q0 * q0 + q1 * q1) - 1
    # r01 = 2 * (q1 * q2 - q0 * q3)
    # r02 = 2 * (q1 * q3 + q0 * q2)
    # r10 = 2 * (q1 * q2 + q0 * q3)
    # r11 = 2 * (q0 * q0 + q2 * q2) - 1
    # r12 = 2 * (q2 * q3 - q0 * q1)
    # r20 = 2 * (q1 * q3 - q0 * q2)
    # r21 = 2 * (q2 * q3 + q0 * q1)
    # r22 = 2 * (q0 * q0 + q3 * q3) - 1
    r00 = - (2 * (q2 * q2 + q3 * q3) - 1)
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = - (2 * (q1 * q1 + q3 * q3) - 1)
    r12 = 2 * (q2 * q3 - q0 * q1)
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = - (2 * (q1 * q1 + q2 * q2) - 1)
    # s = Q[0]
    # x = Q[1]
    # y = Q[2]
    # z = Q[3]
    # r00 = 1 - 2*y**2 - 2*z**2
    # r01 = 2 * (q1 * q2 - q0 * q3)
    # r02 = 2 * (q1 * q3 + q0 * q2)
    # r10 = 2 * (q1 * q2 + q0 * q3)
    # r11 = 2 * (q0 * q0 + q2 * q2) - 1
    # r12 = 2 * (q2 * q3 - q0 * q1)
    # r20 = 2 * (q1 * q3 - q0 * q2)
    # r21 = 2 * (q2 * q3 + q0 * q1)
    # r22 = 2 * (q0 * q0 + q3 * q3) - 1
    return np.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]])

def gen_calibration_matrix(intrinsic, sensor_R, sensor_t, ego_R, ego_t):
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
    intrinsic = np.vstack([intrinsic, [0,0,0]])
    intrinsic = np.hstack([intrinsic, [[0],[0],[0],[1]]])
    return intrinsic, sensor_extrinsic, ego_extrinsic

def coord_3d_to_2d(coord_3d, calibration_matrix, calibrated=True):
    intrinsic, sensor_R, sensor_t, ego_R, ego_t = calibration_matrix['camera_intrinsic'], calibration_matrix['sensor_R'], calibration_matrix['sensor_t'], calibration_matrix['ego_R'], calibration_matrix['ego_t']
    coord_3d, intrinsic, sensor_R, sensor_t, ego_R, ego_t = np.array(coord_3d), np.array(intrinsic), np.array(sensor_R), np.array(sensor_t), np.array(ego_R), np.array(ego_t)
    # change to homogeneous coordinate
    coord_3d = np.append(coord_3d, 1)
    intrinsic, sensor_extrinsic, ego_extrinsic = gen_calibration_matrix(intrinsic, sensor_R, sensor_t, ego_R, ego_t)
    # no need extrinsic matrix because the coordinate has been rotated and translated
    if calibrated:
        coord_2d = intrinsic @ coord_3d
    else:
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
    coord_2d = np.asarray(coord_2d)
    intrinsic_inverse = np.linalg.inv(intrinsic)
    coord_2d = np.append(coord_2d, 1)
    coord_3d = depth * intrinsic_inverse @ coord_2d
    coord_3d = (coord_3d / coord_3d[[-1]])[:3].T
    return coord_3d


    