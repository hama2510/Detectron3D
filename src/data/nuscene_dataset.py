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

def flip_2d_array_horizontally(arr):
    flipped_array = [row[::-1] for row in arr]
    return np.array(flipped_array)

class NusceneDataset(Dataset):
    def __init__(self, data_file, config, return_target=True, is_train=True):
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
        self.aug_config = config.data.aug
        self.aug = config.data.aug.copy()
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def aug_data(self, img, targets):
        if 'flip' in self.aug_config.keys():
            rand = np.random.rand()
            if rand<self.aug_config.flip.rate:
                img = cv.flip(img, 1)
                for stride in targets.keys():
                    for key in targets[stride].keys():
                        targets[stride][key] = flip_2d_array_horizontally(targets[stride][key])
        return img, targets

    def to_float_tensor(self, targets):
        for stride in targets.keys():
            for key in targets[stride].keys():
                targets[stride][key] = torch.FloatTensor(targets[stride][key])
        return targets

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data[idx]
        img = cv.imread(item["image"])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(
            img, (int(img.shape[1] * self.resize), int(img.shape[0] * self.resize))
        )
        shape = [img.shape[0], img.shape[1]]
        raw_img = img.copy()
        
        targets = {}
        if self.return_target:
            for stride in self.stride_list:
                targets["{}".format(stride)] = self.gen_target(
                    item["annotations"], shape, stride
                )
        if self.is_train:
            img, targets = self.aug_data(img, targets)
        img = transforms.Compose([transforms.ToTensor()])(img.copy())
        targets = self.to_float_tensor(targets)
        sample = {
            "sample_token": item["sample_token"],
            "calibration_matrix": item["calibration_matrix"],
            "img_path": item["image"],
            "img": img,
            # "raw_img": raw_img,
            "target": targets,
        }
        return sample

    def is_close_box(self, depth, thres=50):
        if depth<=thres:
            return True
        else:
            return False

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

    def gen_category_onehot(self, category, tag=None):
        category = self.meta_data["category_map"][category.split(".")[-1]]
        if tag=='dc':
            onehot = np.zeros(len(self.meta_data["categories"])) -1
            return onehot
        else:
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

    def gen_target(self, anns, img_shape, stride):
        raise NotImplementedError()