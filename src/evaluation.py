from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import *
from nuscenes.eval.detection.render import visualize_sample
import json
import os
import shutil
from nuscenes.eval.common.loaders import load_gt
from nuscenes.eval.detection.data_classes import DetectionBox
from functools import partial
from multiprocessing import Pool
import numpy as np
import cv2 as cv
from utils.camera import coord_3d_to_2d
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
# def draw_cube(item):
#     cube_edges = [
#         (0, 1), (1, 2), (2, 3), (3, 0),
#         (0, 4), (1, 5), (2, 6), (3, 7),
#         (4, 5), (5, 6), (6, 7), (7, 4)
#     ]
#     box = Box(item['translation'], item['size'], item(item['rotation']))
#     corners = box.corners()
#     corners_2d = np.asarray([coord_3d_to_2d(corners[:,i], calib, calibrated=True) for i in range(corners.shape[1])])
#     for edge in cube_edges:
#         start_point = corners_2d[edge[0]]
#         end_point = corners_2d[edge[1]]
#         img = cv.line(img, start_point, end_point, (0, 0, 255), 1)

class Evaluation:
    def __init__(self, dataset_name, dataroot, eval_config, verbose=False, ):
        self.nusc = NuScenes(version=dataset_name, dataroot=dataroot, verbose=verbose)
#         if 'mini' in dataset_name:
#             self.eval_set = 'mini_val'
        self.cfg_path = eval_config
        # self.output_dir = output_dir
        # self.result_path = os.path.join(self.output_dir, 'result_tmp.json')
        
    def dummy_box(self, sample_token):
        return {
                    'sample_token': sample_token,
                    'translation': [0,0,0],
                    'size': [0,0,0],
                    'rotation': [0,0,0,0],
                    'velocity': [0,0],
                    'detection_name': 'barrier',
                    'detection_score': 0,
                    'attribute_name': '',
                }
    
    def clear(self, output_dir):
         shutil.rmtree(output_dir) 
            
    def evaluate(self, preds, eval_set='val', verbose=False, clear=True, plot_examples=0,conf_th=0.05, output_dir='../tmp/'):
        result_path = os.path.join(output_dir, 'result.json')
        gt_boxes = load_gt(self.nusc, eval_set, DetectionBox, verbose=verbose)
        sample_tokens = set(gt_boxes.sample_tokens)
        result = {"meta":{"use_camera":True},"results":{}}
        for sample_token in sample_tokens:
            boxes = [item for item in preds if item['sample_token']==sample_token]
            boxes.sort(key=lambda x: x["detection_score"])
            if len(boxes) > 500:
                boxes = boxes[:500]
            result['results'][sample_token] = boxes
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        out_file = open(result_path, "w")
        json.dump(result, out_file, cls=NpEncoder)
        out_file.close()
        
        with open(self.cfg_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))
        nusc_eval = DetectionEval(self.nusc, config=cfg_, result_path=result_path, eval_set=eval_set, output_dir=output_dir, verbose=verbose)
        
        metrics, metric_data_list = nusc_eval.evaluate()
        metrics_summary = metrics.serialize()
        with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(nusc_eval.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(output_dir, 'examples')
            for sample_token in sample_tokens:
#                 dir =  os.path.join(example_dir, sample_token)
                os.makedirs(example_dir, exist_ok=True)

                try:
                    visualize_sample(self.nusc,
                                    sample_token,
                                    nusc_eval.gt_boxes if nusc_eval.eval_set != 'test' else EvalBoxes(),
                                    nusc_eval.pred_boxes,
                                    conf_th=conf_th,
                                    eval_range=max(nusc_eval.cfg.class_range.values()),
                                    savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))
                except Exception as e:
                    print(e)

        metrics_summary = json.load(open(os.path.join(output_dir, 'metrics_summary.json'), 'r'))
        if clear:
            self.clear(output_dir) 
        return metrics_summary