import sys, os
import torch

import pickle
import numpy as np

sys.path.append("..")
from utils.camera import *
from .nuscene_dataset import *

STRIDE_LIST = [16]
M_LIST = [0, np.inf]
RADIUS = 1.5


class NusceneDatasetFCOS3D(NusceneDataset):
    def __init__(self, data_file, config, return_target=True):
        super().__init__(data_file, config, return_target)
        self.stride_list = STRIDE_LIST
        self.m_list = M_LIST
        self.radius = RADIUS

    def gen_target(self, anns, img_shape, stride):
        shape = [
            int(np.ceil(img_shape[0] / stride)),
            int(np.ceil(img_shape[1] / stride)),
        ]
        category_target = np.zeros(
            (shape[0], shape[1], len(self.meta_data["categories"]))
        )
        attribute_target = np.zeros(
            (shape[0], shape[1], len(self.meta_data["attributes"]))
        )
        centerness_target = np.zeros((shape[0], shape[1], 1))
        offset_target = np.zeros((shape[0], shape[1], 2))
        depth_target = np.zeros((shape[0], shape[1], 1))
        size_target = np.zeros((shape[0], shape[1], 3))
        rotation_target = np.zeros((shape[0], shape[1], 1))
        dir_target = np.zeros((shape[0], shape[1], 2))
        velocity_target = np.zeros((shape[0], shape[1], 2))

        if self.transformed:
            pass
        else:
            for x in range(shape[1]):
                for y in range(shape[0]):
                    boxes = []
                    for ann in anns:
                        box_2d = [
                            np.array(
                                [
                                    int(ann["box_2d"][0][0] * self.resize),
                                    int(ann["box_2d"][0][1] * self.resize),
                                ]
                            ),
                            int(ann["box_2d"][1] * self.resize),
                            int(ann["box_2d"][2] * self.resize),
                        ]
                        box_2d = np.asarray(box_2d, dtype=object)
                        # pass_cond = is_near_center([x, y],  box_2d//stride)
                        pass_cond = is_center([x, y], box_2d // stride)
                        # pass_cond = is_positive_location(
                        #     [x, y], box_2d, stride, self.radius
                        # )
                        pass_cond = pass_cond and is_valid_box(
                            box_2d, (img_shape[1], img_shape[0])
                        )
                        # pass_cond = pass_cond and check_box_and_feature_map_level([x, y], ann['box_2d'], stride, self.m_list, self.stride_list)
                        if pass_cond:
                            new_ann = ann.copy()
                            new_ann["box_2d"] = box_2d
                            boxes.append(new_ann)
                    if len(boxes) > 0:
                        # foreground location
                        boxes.sort(
                            key=lambda item: distance_to_center(
                                [
                                    x * stride,
                                    y * stride,
                                ],
                                item["box_2d"],
                            )
                        )
                        box = boxes[0]
                        box_2d = np.asarray(box["box_2d"], dtype=object) // stride
                        if self.rotation_encode == "sin_pi_and_bin":
                            rad, dir_cls = self.rotation_angle_to_sin_pi_and_bin(
                                ann["yaw_angle_rad"]
                            )
                        elif self.rotation_encode == "pi_and_minus_pi":
                            rad = self.rotation_angle_to_pi_and_minus_pi(
                                ann["yaw_angle_rad"]
                            )
                            dir_cls = 0

                        category_onehot = self.gen_category_onehot(box["category"])
                        if category_onehot is None:
                            # skip void objects
                            continue

                        category_target[y, x, :] = category_onehot
                        attribute_target[y, x, :] = self.gen_attribute_onehot(
                            box["attribute"]
                        )
                        centerness_target[y, x, :] = self.centerness([x, y], box_2d)
                        offset_target[y, x, :] = self.offset(
                            [x, y], box["box_2d"], stride
                        )
                        depth_target[y, x, :] = box["xyz_in_sensor_coor"][2]
                        size_target[y, x, :] = box["box_size"]
                        rotation_target[y, x, :] = rad
                        dir_target[y, x, :] = dir_cls
                        velocity_target[y, x, :] = self.gen_velocity(box["velocity"])

        return {
            "category": category_target,
            "attribute": attribute_target,
            "centerness": centerness_target,
            "offset": offset_target,
            "depth": depth_target,
            "size": size_target,
            "rotation": rotation_target,
            "dir": dir_target,
            "velocity": velocity_target,
        }
