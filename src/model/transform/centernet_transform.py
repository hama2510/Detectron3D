import numpy as np
from .base_transform import BaseTransformer


class CenterNetTransformer(BaseTransformer):
    def transform_predict(self, pred, det_thres=0.5):
        boxes = []
        sample_token = pred["sample_token"]
        calib_matrix = pred["calibration_matrix"]
        img_path = pred["img_path"]
        for key in pred["pred"].keys():
            stride = 2 ** int(key[1:])
            category_map = pred["pred"][key]["category"]
            attribute_map = pred["pred"][key]["attribute"]
            offset_map = pred["pred"][key]["offset"]
            depth_map = pred["pred"][key]["depth"]
            size_map = pred["pred"][key]["size"]
            rotation_map = pred["pred"][key]["rotation"]
            velocity_map = pred["pred"][key]["velocity"]
            pred_score = np.max(category_map, axis=2)
            indices = np.argwhere(pred_score > det_thres)
            sorted_indices = indices[np.argsort(indices.sum(axis=1))]
            if topk>0:
                sorted_indices = sorted_indices[:topk]
                
            for idx in sorted_indices:
                sc = pred_score[idx[0], idx[1]]
                x, y = self.gen_coord_from_map(idx, offset_map, stride)
                depth = np.exp(depth_map[idx[0]][idx[1], 0])
                size = np.clip(size_map[idx[0], idx[1], :], a_min=1e-4, a_max=None)

                if self.config.data.rotation_encode == "pi_and_minus_pi":
                    rotation = rotation_map[idx[0], idx[1], 0] * np.pi * 2.0
                else:
                    raise ValueError('Not support other than pi_and_minus_pi')

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

    def transform_target(self, pred, det_thres=0.05, topk=-1):
        boxes = []
        sample_token = pred["sample_token"]
        calib_matrix = pred["calibration_matrix"]
        img_path = pred["img_path"]
        for key in pred["pred"].keys():
            stride = int(key)
            category_map = pred["pred"][key]["category"]
            attribute_map = pred["pred"][key]["attribute"]
            offset_map = pred["pred"][key]["offset"]
            depth_map = pred["pred"][key]["depth"]
            size_map = pred["pred"][key]["size"]
            rotation_map = pred["pred"][key]["rotation"]
            velocity_map = pred["pred"][key]["velocity"]
            pred_score = np.max(category_map, axis=2)
            indices = np.argwhere(pred_score > det_thres)

            for idx in indices:
                sc = pred_score[idx[0], idx[1]]
                x, y = self.gen_coord_from_map(idx, offset_map, stride)
                depth = depth_map[idx[0]][idx[1], 0]
                size = np.clip(size_map[idx[0], idx[1], :], a_min=1e-4, a_max=None)

                if self.config.data.rotation_encode == "pi_and_minus_pi":
                    rotation = rotation_map[idx[0], idx[1], 0] * np.pi * 2.0
                else:
                    raise ValueError('Not support other than pi_and_minus_pi')
                
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
