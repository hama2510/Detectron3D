import numpy as np
import pickle
from pyquaternion import Quaternion
import sys, os
import scipy

sys.path.append("..")
from utils.nms import rotated_nms
from utils.camera import coord_2d_to_3d, sensor_coord_to_real_coord
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from datetime import datetime


class BaseTransformer:
    def __init__(self, config):
        self.config = config
        self.meta_data = pickle.load(open(config.data.meta_data, "rb"))

    def gen_box(
        self,
        sample_token,
        calib_matrix,
        img_path,
        coord,
        depth,
        rotation,
        size,
        category,
        attribute,
        score,
        velocity,
    ):
        category = self.meta_data["categories"][category]
        if category in ["barrier", "traffic_cone"]:
            attribute = ""
        else:
            attribute = self.meta_data["attributes"][attribute]
            if attribute == "void":
                attribute = ""
        coord_3d = coord_2d_to_3d(coord, depth, calib_matrix)
        rotation_q = Quaternion(axis=[0, 0, 1], angle=rotation)
        box_real = sensor_coord_to_real_coord(
            coord_3d, size, rotation_q, calib_matrix, rotate_yaw_only=True
        )
        return {
            "sample_token": sample_token,
            "translation": box_real.center,
            "size": box_real.wlh,
            "rotation": box_real.orientation.elements,
            "rotation_angle": box_real.orientation.angle,
            "velocity": velocity,
            "detection_name": category,
            "detection_score": score,
            "attribute_name": attribute,
            "img_path": img_path,
        }

    def transform_predict(self):
        raise NotImplementedError()

    def transform_target(self):
        raise NotImplementedError()
    
    def gen_coord_from_map(self, idx, offset_map, stride):
        y = int(idx[0] * stride + offset_map[idx[0], idx[1], 1])
        x = int(idx[1] * stride + offset_map[idx[0], idx[1], 0])
        x = int(x / self.config.data.resize)
        y = int(y / self.config.data.resize)
        return x, y

    def transform_predicts(self, preds, target=False):
        boxes = []
        if self.config.num_workers <= 1:
            for pred in preds:
                boxes.extend(
                    self.transform_predict(pred, det_thres=self.config.det_thres)
                )
        else:
            if not target:
                func = self.transform_predict
            else:
                func = self.transform_target
            start = datetime.now()
            pool = Pool(self.config.num_workers)
            data = list(
                pool.imap(
                    partial(func, det_thres=self.config.det_thres),
                    preds,
                )
            )
            pool.close()
            pool.join()
            print("Transforming prediction at ", datetime.now() - start)
            start = datetime.now()
            total_box = sum([len(item) for item, _ in data])
            for item, calib_matrix in data:
                # keep_indices = list(range(len(item)))
                keep_indices = rotated_nms(
                    item, calib_matrix, nms_thres=self.config.nms_thres
                )
                boxes.extend([item[i] for i in keep_indices])
            print(
                "Running NMS from {} to {} at ".format(total_box, len(boxes)),
                datetime.now() - start,
            )
        return boxes
