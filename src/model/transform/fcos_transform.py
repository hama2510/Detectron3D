import numpy as np
import scipy
from .base_transform import BaseTransformer


class FCOSTransformer(BaseTransformer):
    def transform_predict(self, pred, det_thres=0.05, **kwargs):
        boxes = []
        sample_token = pred["sample_token"]
        calib_matrix = pred["calibration_matrix"]
        img_path = pred["img_path"]
        for key in pred["pred"].keys():
            stride = 2 ** int(key[1:])
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

            for idx in indices:
                sc = pred_score[idx[0], idx[1]]
                y = int(idx[0] * stride + offset_map[idx[0], idx[1], 1])
                x = int(idx[1] * stride + offset_map[idx[0], idx[1], 0])
                x = int(x / self.config.data.resize)
                y = int(y / self.config.data.resize)
                depth = np.exp(depth_map[idx[0]][idx[1], 0])
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

                box = self.gen_box(
                    sample_token,
                    calib_matrix,
                    img_path,
                    [x, y],
                    depth,
                    rotation,
                    size,
                    np.argmax(category_map[idx[0], idx[1], :]),
                    np.argmax(attribute_map[idx[0], idx[1], :]),
                    sc,
                    velocity_map[idx[0], idx[1], :],
                )
                boxes.append(box)
        return boxes, calib_matrix

    def transform_target(self, pred, det_thres=0.05, **kwargs):
        boxes = []
        sample_token = pred["sample_token"]
        calib_matrix = pred["calibration_matrix"]
        img_path = pred["img_path"]
        for key in pred["pred"].keys():
            stride = int(key)
            category_map = pred["pred"][key]["category"]
            attribute_map = pred["pred"][key]["attribute"]
            centerness_map = pred["pred"][key]["centerness"]
            offset_map = pred["pred"][key]["offset"]
            depth_map = pred["pred"][key]["depth"]
            size_map = pred["pred"][key]["size"]
            rotation_map = pred["pred"][key]["rotation"]
            dir_map = scipy.special.softmax(pred["pred"][key]["dir"], axis=2)
            velocity_map = pred["pred"][key]["velocity"]
            cls_score = np.max(category_map, axis=2)
            pred_score = cls_score * centerness_map[:, :, 0]
            indices = np.argwhere(pred_score > det_thres)

            for idx in indices:
                sc = pred_score[idx[0], idx[1]]
                y = int(idx[0] * stride + offset_map[idx[0], idx[1], 1])
                x = int(idx[1] * stride + offset_map[idx[0], idx[1], 0])
                x = int(x / self.config.data.resize)
                y = int(y / self.config.data.resize)
                depth = depth_map[idx[0]][idx[1], 0]
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
                box = self.gen_box(
                    sample_token,
                    calib_matrix,
                    img_path,
                    [x, y],
                    depth,
                    rotation,
                    size,
                    np.argmax(category_map[idx[0], idx[1], :]),
                    np.argmax(attribute_map[idx[0], idx[1], :]),
                    sc,
                    velocity_map[idx[0], idx[1], :],
                )
                boxes.append(box)
        return boxes, calib_matrix
