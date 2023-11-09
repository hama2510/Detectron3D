import numpy as np
from .centernet_transform import CenterNetTransformer
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from datetime import datetime

class CenterNetTransformerDepth(CenterNetTransformer):

    def find_neighbors(self, coord, shape, n=1):
        x, y = coord
        max_x, max_y = shape
        neighbors = []

        for i in range(x-n, x+n+1):
            for j in range(y-n, y+n+1):
                if i>=0 and i<max_x and j>=0 and j<max_y and (x!=i or y!=j):
                    neighbors.append([i, j])
        return neighbors

    def transform_predict(self, pred, det_thres=0.5, topk=-1):
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
            score_list = []
            for x in range(pred_score.shape[0]):
                for y in range(pred_score.shape[1]):
                    sc = pred_score[x, y]
                    if sc>det_thres:
                        neighbors = self.find_neighbors((x,y), pred_score.shape)
                        if all([sc>pred_score[nx, ny] for (nx, ny) in neighbors]):
                            score_list.append([[x, y], sc])
            score_list.sort(key=lambda x: x[1], reverse=True)
            if topk > 0:
                score_list = score_list[:topk]

            for idx, sc in score_list:
                x, y = self.gen_coord_from_map(idx, offset_map, stride)
#                 depth = np.argmax(depth_map[idx[0]][idx[1], :])
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
                    np.argmax(depth_map[idx[0]][idx[1], :])+1,
                    rotation,
                    size,
                    np.argmax(category_map[idx[0], idx[1], :]),
                    np.argmax(attribute_map[idx[0], idx[1], :]),
                    sc,
                    velocity_map[idx[0], idx[1], :],
                )
                boxes.append(box)
        return boxes, calib_matrix

    
    def transform_predicts(self, preds, target=False):
        boxes = []
        if self.config.num_workers <= 1:
            for pred in preds:
                boxes.extend(
                    self.transform_predict(pred, det_thres=self.config.demo.det_thres, topk=self.config.demo.topk)
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
                    partial(func, det_thres=self.config.demo.det_thres, topk=self.config.demo.topk),
                    preds,
                )
            )
            pool.close()
            pool.join()
            print("Transforming prediction at ", datetime.now() - start)
            for item, calib_matrix in data:
                boxes.extend(item)
            print('Total {} boxes'.format(len(boxes)))
        return boxes