import sys, os
import torch

import pickle
import numpy as np

sys.path.append("..")
from utils.camera import *
from .nuscene_dataset_centernet import *

STRIDE_LIST = [16]
M_LIST = [0, np.inf]
RADIUS = 1.5
DEPTH_THRES = 50
NUM_DEPTH = 50


class NusceneDatasetCenterNetDepth(NusceneDatasetCenterNet):

    def gen_depth(self, depth, num=50, r=50):
        n = r//num
        idx = int(depth//n)-1
        idx = np.max([0, idx])
        onehot = np.zeros(num)
        onehot[idx] = 1
        return onehot

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
        offset_target = np.zeros((shape[0], shape[1], 2))
        depth_target = np.zeros((shape[0], shape[1], 50))
        size_target = np.zeros((shape[0], shape[1], 3))
        rotation_target = np.zeros((shape[0], shape[1], 1))
        velocity_target = np.zeros((shape[0], shape[1], 2))

        if self.transformed:
            pass
        else:
            for x in range(shape[1]):
                for y in range(shape[0]):
                    boxes = []
                    for ann in anns:
                        box_2d = np.array(
                            [
                                np.array(
                                    [
                                        int(ann["box_2d"][0][0] * self.resize),
                                        int(ann["box_2d"][0][1] * self.resize),
                                    ]
                                ),
                                int(ann["box_2d"][1] * self.resize),
                                int(ann["box_2d"][2] * self.resize),
                            ], dtype=object
                        )
                        pass_cond = is_center([x, y], box_2d // stride)
                        pass_cond = pass_cond and is_valid_box(
                            box_2d, (img_shape[1], img_shape[0])
                        )
                        pass_cond = pass_cond and self.is_close_box(ann["xyz_in_sensor_coor"][2], thres=DEPTH_THRES)
                        if pass_cond:
                            new_ann = ann.copy()
                            new_ann["box_2d"] = box_2d
                            boxes.append(new_ann)
                    if len(boxes) > 0:
                        # foreground location
                        boxes.sort(
                            key=lambda item: distance_to_center(
                                [
                                    x * stride + np.floor(stride / 2),
                                    y * stride + np.floor(stride / 2),
                                ],
                                item["box_2d"],
                            )
                        )
                        box = boxes[0]
                        # box_2d = np.asarray(box["box_2d"], dtype=object) // stride
                        rad = self.rotation_angle_to_pi_and_minus_pi(
                            box["yaw_angle_rad"]
                        )

                        category_onehot = self.gen_category_onehot(box["category"], box['tag'])
                        if category_onehot is None:
                            # skip void objects
                            continue

                        category_target[y, x, :] = category_onehot
                        attribute_target[y, x, :] = self.gen_attribute_onehot(
                            box["attribute"]
                        )
                        offset_target[y, x, :] = self.offset([x, y], box["box_2d"], stride)
                        depth_target[y, x, :] = self.gen_depth(box["xyz_in_sensor_coor"][2], num=DEPTH_THRES)
                        size_target[y, x, :] = box["box_size"]
                        rotation_target[y, x, :] = rad
                        velocity_target[y, x, :] = self.gen_velocity(box["velocity"])

        return {
            "category": category_target,
            "attribute": attribute_target,
            "offset": offset_target,
            "depth": depth_target,
            "size": size_target,
            "rotation": rotation_target,
            "velocity": velocity_target,
        }
