import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from .fcos3d import FCOS3D
from .fcos3d_fused import FCOS3DFused, FCOS3DFusedP3
from .mobilenet_v2 import MobileNetv2
from .resnet101 import ResNet101
from .resnet101_deformable import ResNet101DCN
import pickle
from pyquaternion import Quaternion
import sys, os

sys.path.append("..")
from utils.nms import rotated_nms
from utils.camera import coord_2d_to_3d, sensor_coord_to_real_coord
from datetime import datetime
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
from datetime import datetime


class FCOSDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.meta_data = pickle.load(open(config.data.meta_data, "rb"))
        self.init()

    def forward(self, x):
        return self.model(x)

    def init(
        self,
    ):
        self.model = self.create_model()
        if "load_model" in self.config.model.keys() and self.config.model.load_model:
            self.load_model(self.config.model.load_model)
            print("Loaded weight from {}".format(self.config.model.load_model))
        if "multi_gpu" in self.config.keys() and self.config.multi_gpu:
            if "gpus" in self.config:
                #                 self.model = DistributedDataParallel(self.model, device_ids=self.config.gpus)
                self.model = nn.DataParallel(self.model, device_ids=self.config.gpus)
            else:
                #                 self.model = DistributedDataParallel(self.model)
                self.model = nn.DataParallel(self.model)
        if self.config.model["eval"]:
            self.model.eval()

    def create_model(
        self,
    ):
        if self.config.model.detector_name == "fcos3d":
            detector = FCOS3D
        elif self.config.model.detector_name == "fcos3d_fused":
            detector = FCOS3DFused
        elif self.config.model.detector_name == "fcos3d_fused_p3":
            detector = FCOS3DFusedP3
        if self.config.model.model_name == "mobilenet":
            model = detector(
                feature_extractor=MobileNetv2(self.config.device, pretrained=True),
                num_cate=len(self.meta_data["categories"]),
                num_attr=len(self.meta_data["attributes"]),
            )
        # elif 'efficientnet' in self.config.model.model_name:
        #     model = EfficientNet.from_pretrained(self.config.model.model_name, num_classes=self.config.model.num_class)
        elif self.config.model.model_name == "resnet101":
            model = detector(
                feature_extractor=ResNet101(self.config.device, pretrained=True),
                num_cate=len(self.meta_data["categories"]),
                num_attr=len(self.meta_data["attributes"]),
            )
        elif self.config.model.model_name == "resnet101_dcn":
            model = detector(
                feature_extractor=ResNet101DCN(),
                num_cate=len(self.meta_data["categories"]),
                num_attr=len(self.meta_data["attributes"]),
            )
        else:
            print("Not support model {}".format(config.model.model_name))
            exit()
        model.to(self.config["device"])
        return model

    def save_model(self, path):
        new_state_dict = OrderedDict()
        if self.config.multi_gpu:
            torch.save(self.model.module.state_dict(), path)
        #             for k, v in self.model.module.state_dict().items():
        #                 new_state_dict[k] = v
        else:
            torch.save(self.model.state_dict(), path)

    #             for k, v in self.model.state_dict().items():
    #                 new_state_dict[k] = v
    #         torch.save(new_state_dict, path)

    def load_model(self, path):
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

    def item_tensor_to_numpy(self, key, item):
        if key == "category":
            item = torch.clamp(item, min=1e-4, max=1 - 1e-4)
        elif key == "attribute" or key == "dir":
            item = nn.functional.softmax(item, dim=1)
        item = item.detach().cpu().numpy()
        item = np.moveaxis(item, 0, -1)
        return item

    def tensor_to_numpy(self, pred):
        output = {
            "sample_token": pred["sample_token"],
            "calibration_matrix": pred["calibration_matrix"],
            "pred": {},
        }
        for key in pred["pred"].keys():
            output["pred"][key] = {}
            category_map = (
                torch.clamp(pred["pred"][key]["category"], min=0, max=1)
                .detach()
                .cpu()
                .numpy()
            )
            attribute_map = (
                nn.functional.softmax(pred["pred"][key]["attribute"], dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            #             attribute_map = pred['pred'][key]['attribute'].detach().cpu().numpy()
            centerness_map = pred["pred"][key]["centerness"].detach().cpu().numpy()
            offset_map = pred["pred"][key]["offset"].detach().cpu().numpy()
            depth_map = pred["pred"][key]["depth"].detach().cpu().numpy()
            size_map = pred["pred"][key]["size"].detach().cpu().numpy()
            rotation_map = pred["pred"][key]["rotation"].detach().cpu().numpy()
            dir_map = (
                nn.functional.softmax(pred["pred"][key]["dir"], dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            #             dir_map = pred['pred'][key]['dir'].detach().cpu().numpy()
            velocity_map = pred["pred"][key]["velocity"].detach().cpu().numpy()

            category_map = np.moveaxis(category_map, 0, -1)
            attribute_map = np.moveaxis(attribute_map, 0, -1)
            centerness_map = np.moveaxis(centerness_map, 0, -1)
            offset_map = np.moveaxis(offset_map, 0, -1)
            depth_map = np.moveaxis(depth_map, 0, -1)
            size_map = np.moveaxis(size_map, 0, -1)
            rotation_map = np.moveaxis(rotation_map, 0, -1)
            dir_map = np.moveaxis(dir_map, 0, -1)
            velocity_map = np.moveaxis(velocity_map, 0, -1)

            output["pred"][key]["category"] = category_map
            output["pred"][key]["attribute"] = attribute_map
            output["pred"][key]["centerness"] = centerness_map
            output["pred"][key]["offset"] = offset_map
            output["pred"][key]["depth"] = depth_map
            output["pred"][key]["size"] = size_map
            output["pred"][key]["rotation"] = rotation_map
            output["pred"][key]["dir"] = dir_map
            output["pred"][key]["velocity"] = velocity_map
        return output


class FCOSTransformer:
    def __init__(self, config):
        self.config = config
        self.meta_data = pickle.load(open(config.data.meta_data, "rb"))

    def transform_predict(self, pred, det_thres=0.05, nms_thres=0.3):
        boxes = []
        sample_token = pred["sample_token"]
        calib_matrix = pred["calibration_matrix"]
        #         stride_list = [32, 64]
        for key in pred["pred"].keys():
            #             stride = int(key)
            stride = 2 ** int(key[1:])
            #             if not stride in stride_list:
            #                 continue
            category_map = pred["pred"][key]["category"]
            attribute_map = pred["pred"][key]["attribute"]
            centerness_map = pred["pred"][key]["centerness"]
            offset_map = pred["pred"][key]["offset"]
            depth_map = pred["pred"][key]["depth"]
            size_map = pred["pred"][key]["size"]
            rotation_map = pred["pred"][key]["rotation"]
            dir_map = pred["pred"][key]["dir"]
            velocity_map = pred["pred"][key]["velocity"]
            cls_score = np.max(category_map, axis=2)
            pred_score = cls_score * centerness_map[:, :, 0]
            indices = np.argwhere(pred_score > det_thres)
            #             indices = np.unique(indices, axis=0)
            for idx in indices:
                sc = pred_score[idx[0], idx[1]]
                #                 y, x = int(idx[0]*stride+offset_map[idx[0], idx[1],0]), int(idx[1]*stride+offset_map[idx[0], idx[1],1])
                y = int(idx[0] + offset_map[idx[0], idx[1], 0]) * stride + np.floor(
                    stride
                )
                x = int(idx[1] + offset_map[idx[0], idx[1], 1]) * stride + np.floor(
                    stride
                )
                x = int(x / self.config.data.resize)
                y = int(y / self.config.data.resize)
                depth = np.exp(depth_map[idx[0]][idx[1], 0])
                #                 depth = depth_map[idx[0],idx[1],0]
                coord_3d = coord_2d_to_3d([x, y], depth, calib_matrix)
                size = np.clip(size_map[idx[0], idx[1], :], a_min=1e-4, a_max=None)

                if self.config.data.rotation_encode == "pi_and_minus_pi":
                    rotation = rotation_map[idx[0], idx[1], 0] * np.pi * 2.0
                elif self.config.data.rotation_encode == "sin_pi_and_bin":
                    rotation = np.arcsin(rotation_map[idx[0], idx[1], 0])
                    dir = np.argmax(dir_map[idx[0], idx[1], :])
                    if dir == 0:
                        if rotation > 0:
                            rotation += np.pi / 2
                        else:
                            rotation -= np.pi / 2
                rotation_q = Quaternion(axis=[0, 0, 1], angle=rotation)
                velocity = velocity_map[idx[0], idx[1], :]
                category = self.meta_data["categories"][
                    np.argmax(category_map[idx[0], idx[1], :])
                ]
                if category in ["barrier", "traffic_cone"]:
                    attribute = ""
                else:
                    attribute = self.meta_data["attributes"][
                        np.argmax(attribute_map[idx[0], idx[1], :])
                    ]
                    if attribute == "void":
                        attribute = ""

                boxes.append(
                    {
                        "sample_token": sample_token,
                        "translation": sensor_coord_to_real_coord(
                            coord_3d, size, rotation_q, calib_matrix
                        ),
                        "size": size,
                        "rotation": rotation_q.elements,
                        "rotation_angle": rotation,
                        "velocity": velocity,
                        "detection_name": category,
                        "detection_score": sc,
                        "attribute_name": attribute,
                    }
                )
        #         keep_indices = rotated_nms(boxes, calib_matrix, nms_thres=nms_thres)
        #         boxes = [boxes[i] for i in keep_indices]
        return boxes, calib_matrix

    def transform_predicts(self, preds):
        boxes = []
        if self.config.num_workers <= 1:
            for pred in preds:
                boxes.extend(
                    self.transform_predict(pred, det_thres=self.config.det_thres)
                )
        else:
            start = datetime.now()
            pool = Pool(self.config.num_workers)
            data = list(
                pool.imap(
                    partial(
                        self.transform_predict,
                        det_thres=self.config.det_thres,
                        nms_thres=self.config.nms_thres,
                    ),
                    preds,
                )
            )
            pool.close()
            pool.join()
            print("Transforming prediction at ", datetime.now() - start)
            start = datetime.now()
            total_box = sum([len(item) for item, calib_matrix in data])
            for item, calib_matrix in data:
                keep_indices = rotated_nms(
                    item, calib_matrix, nms_thres=self.config.nms_thres
                )
                boxes.extend([item[i] for i in keep_indices])
            print("Running NMS from {} to {} at ".format(total_box, len(boxes)), datetime.now() - start)
        return boxes
