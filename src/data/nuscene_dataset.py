import sys, os
import torch

# import pandas as pd
# from skimage import io, transform
import pickle
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append("..")
from utils.camera import *
from tqdm import tqdm
import argparse
import imagesize
from functools import partial
from multiprocessing import Pool

STRIDE_LIST = [16]
# STRIDE_LIST = [8, 16, 32, 64, 128]
# M_LIST = [0, 64, 128, 256, 512, np.inf]
M_LIST = [0, np.inf]
RADIUS = 1.5


def check_box_and_feature_map_level(point, box, stride, m_list, stride_list):
    point = [
        point[0] * stride + np.floor(stride / 2),
        point[1] * stride + np.floor(stride / 2),
    ]
    (x1, y1), (x2, y2) = xywh_to_xyxy(box)
    l = point[0] - x1
    t = point[1] - y1
    r = x2 - point[0]
    b = y2 - point[1]
    m = np.max([l, t, b, r])
    if (
        m < m_list[stride_list.index(stride)]
        or m > m_list[stride_list.index(stride) + 1]
    ):
        return False
    else:
        return True


def is_valid_box(box, shape):
    (x, y), w, h = box
    # (x1, y1), (x2, y2) = xywh_to_xyxy(box)
    if (
        x < 0
        or x > shape[0]
        or y < 0
        or y > shape[1]
        #     x1 < 0
        #     or x2 < 0
        #     or y1 < 0
        #     or y2 < 0
        #     or x1 > shape[0]
        #     or x2 > shape[0]
        #     or y1 > shape[1]
        #     or y2 > shape[1]
    ):
        return False
    else:
        return True


def is_center(point, box):
    if point[0] - box[0][0] == 0 and point[1] - box[0][1] == 0:
        return True
    return False


def is_near_center(point, box, thres=None):
    if thres == None:
        thres = (box[0][1] // 4, box[0][1] // 4)
    if point[0] - box[0][0] <= thres[0] and point[1] - box[0][1] <= thres[1]:
        return True
    return False


def is_positive_location(point, box, stride, radius):
    box_center = [box[0][0] // stride, box[0][1] // stride]
    box_strided = [box_center, box[1] // stride, box[2] // stride]
    d = np.sqrt((box_center[0] - point[0]) ** 2 + (box_center[1] - point[1]) ** 2)
    (x1, y1), (x2, y2) = xywh_to_xyxy(box_strided)
    #     if d<radius*stride and point[0]>x1 and point[0]<x2 and point[1]>y1 and point[1]<y2:
    #         return True
    #     else:
    #         return False
    if point[0] > x1 and point[0] < x2 and point[1] > y1 and point[1] < y2:
        return True
    else:
        return False


# class NusceneDatasetTransform:
#     def __init__(self, num_worker):
#         self.stride_list = STRIDE_LIST
#         self.m_list = M_LIST
#         self.radius = RADIUS
#         self.num_worker = num_worker

#     def update_annotation(self, item):
#         shape = imagesize.get(item["image"])
#         calib_matrix = item["calibration_matrix"]
#         anns = item["annotations"]
#         for stride in self.stride_list:
#             new_shape = [
#                 int(np.ceil(shape[0] / stride)),
#                 int(np.ceil(shape[1] / stride)),
#             ]
#             for ann in item["annotations"]:
#                 if not "targets" in ann.keys():
#                     ann["targets"] = {}
#                 ann["targets"][stride] = []
#             for x in range(new_shape[0]):
#                 for y in range(new_shape[1]):
#                     boxes = []
#                     for idx, ann in enumerate(anns):
#                         if (
#                             is_valid_box(ann["box_2d"], shape)
#                             and is_positive_location(
#                                 [x, y], ann["box_2d"], stride, self.radius
#                             )
#                             and check_box_and_feature_map_level(
#                                 [x, y],
#                                 ann["box_2d"],
#                                 stride,
#                                 self.m_list,
#                                 self.stride_list,
#                             )
#                         ):
#                             boxes.append([idx, ann])
#                     if len(boxes) > 0:
#                         boxes.sort(
#                             key=lambda item: distance_to_center(
#                                 [
#                                     x * stride + np.floor(stride / 2),
#                                     y * stride + np.floor(stride / 2),
#                                 ],
#                                 item[1]["box_2d"],
#                             )
#                         )
#                         idx = boxes[0][0]
#                         item["annotations"][idx]["targets"][stride].append([x, y])
#         return item

#     def transform(self, data, out):
#         data = pickle.load(open(data, "rb"))
#         #         data = data[:10]

#         if self.num_worker > 0:
#             pool = Pool(self.num_worker)
#             r = list(tqdm(pool.imap(self.update_annotation, data), total=len(data)))
#             pool.close()
#             pool.join()
#         else:
#             for item in tqdm(data):
#                 item["annotation"] = self.update_annotation(item)
#             r = data

#         os.makedirs(os.path.dirname(out), exist_ok=True)
#         pickle.dump(r, open(out, "wb"))


class NusceneDataset(Dataset):
    def __init__(self, data_file, config, return_target=True):
        self.transformed = config.data.transformed
        self.image_root = config.data.image_root
        if not self.image_root.endswith("/"):
            self.image_root += "/"
        self.data = pickle.load(open(data_file, "rb"))
        self.meta_data = pickle.load(open(config.data.meta_data, "rb"))
        self.stride_list = STRIDE_LIST
        self.m_list = M_LIST
        self.radius = RADIUS
        self.return_target = return_target
        self.resize = config.data.resize
        self.rotation_encode = config.data.rotation_encode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data[idx]
        #         img = cv.imread(self.image_root+item['image'])
        img = cv.imread(item["image"])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(
            img, (int(img.shape[1] * self.resize), int(img.shape[0] * self.resize))
        )
        shape = [img.shape[0], img.shape[1]]
        raw_img = img.copy()
        img = transforms.Compose([transforms.ToTensor()])(img.copy())
        sample = {
            "sample_token": item["sample_token"],
            "calibration_matrix": item["calibration_matrix"],
            "img_path": item["image"],
            "img": img,
            # "raw_img": raw_img,
            "target": {},
        }
        if self.return_target:
            for stride in self.stride_list:
                sample["target"]["{}".format(stride)] = self.gen_target(
                    item["annotations"], shape, stride, item["calibration_matrix"]
                )
        return sample

    #     def rotation_angle_to_pi_and_bin(self, rotation_angle):
    #         rad, bin = rotation_angle/np.pi, np.max([0, int(np.sign(rotation_angle))])
    #         dir_cls = np.zeros(2)
    #         dir_cls[bin] = 1
    #         return rad, dir_cls

    def rotation_angle_to_sin_pi_and_bin(self, rotation_angle):
        rad = np.sin(rotation_angle)
        dir_cls = np.zeros(2)
        if np.abs(rotation_angle) > np.pi / 2:
            dir_cls[0] = 1
        else:
            dir_cls[1] = 1
        return rad, dir_cls

    def rotation_angle_to_pi_and_minus_pi(self, rotation_angle):
        return rotation_angle / 2.0 / np.pi

    def gen_category_onehot(self, category):
        category = self.meta_data["category_map"][category.split(".")[-1]]
        if category == "void":
            return None
        else:
            onehot = np.zeros(len(self.meta_data["categories"]))
            onehot[self.meta_data["categories"].index(category)] = 1
            return onehot

    def gen_attribute_onehot(self, attribute):
        onehot = np.zeros(len(self.meta_data["attributes"]))
        if len(attribute) == 0:
            onehot[self.meta_data["attributes"].index("void")]
        else:
            onehot[self.meta_data["attributes"].index(attribute[0])] = 1
        return onehot

    def gen_velocity(self, velocity):
        if np.isnan(velocity).any():
            #             not a moving object
            return [0, 0]
        else:
            return [velocity[0], velocity[1]]

    def centerness(self, point, box, alpha=2.5):
        c = np.exp(-alpha * ((point[0] - box[0][0]) ** 2 + (point[1] - box[0][1]) ** 2))
        return c

    def offset(self, point, box, stride):
        # return [box[0][0] - point[0], box[0][1] - point[1]]

        return [box[0][0] - point[0] * stride, box[0][1] - point[1] * stride]

    def gen_target(self, anns, img_shape, stride, calib_matrix):
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
            # for ann in anns:
            #     for x, y in ann["targets"][stride]:
            #         box_2d = (
            #             np.asarray(ann["box_2d"], dtype=object) // stride * self.resize
            #         )
            #         if self.rotation_encode == "sin_pi_and_bin":
            #             rad, dir_cls = self.rotation_angle_to_sin_pi_and_bin(
            #                 ann["rotation_angle_rad"]
            #             )
            #         elif self.rotation_encode == "pi_and_minus_pi":
            #             rad = self.rotation_angle_to_pi_and_minus_pi(
            #                 ann["rotation_angle_rad"]
            #             )
            #             dir_cls = 0

            #         category_onehot = self.gen_category_onehot(ann["category"])
            #         if category_onehot is None:
            #             # skip void objects
            #             continue
            #         category_target[y, x, :] = category_onehot
            #         attribute_target[y, x, :] = self.gen_attribute_onehot(
            #             ann["attribute"]
            #         )
            #         centerness_target[y, x, :] = self.centerness([x, y], box_2d)
            #         offset_target[y, x, :] = self.offset([x, y], box_2d)
            #         depth_target[y, x, :] = ann["xyz_in_sensor_coor"][2]
            #         size_target[y, x, :] = ann["box_size"]
            #         rotation_target[y, x, :] = rad
            #         dir_target[y, x, :] = dir_cls
            #         velocity_target[y, x, :] = self.gen_velocity(ann["velocity"])
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
                                ann["rotation_angle_rad"]
                            )
                        elif self.rotation_encode == "pi_and_minus_pi":
                            rad = self.rotation_angle_to_pi_and_minus_pi(
                                ann["rotation_angle_rad"]
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
            "category": torch.FloatTensor(category_target),
            "attribute": torch.FloatTensor(attribute_target),
            "centerness": torch.FloatTensor(centerness_target),
            "offset": torch.FloatTensor(offset_target),
            "depth": torch.FloatTensor(depth_target),
            "size": torch.FloatTensor(size_target),
            "rotation": torch.FloatTensor(rotation_target),
            "dir": torch.FloatTensor(dir_target),
            "velocity": torch.FloatTensor(velocity_target),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--file", type=str, help="path_to_in_file")
    parser.add_argument("--out", type=str, help="path_to_out_file")
    parser.add_argument("--num_worker", type=int, default=16, help="num_worker")
    args = parser.parse_args()

    transform = NusceneDatasetTransform(num_worker=args.num_worker)
    transform.transform(args.file, args.out)
