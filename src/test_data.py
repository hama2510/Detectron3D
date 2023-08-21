import pandas as pd
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.nuscene_dataset import NusceneDataset
from model.fcos3d_detector import FCOSDetector, FCOSTransformer
from criterion.losses import Criterion
import argparse
from omegaconf import OmegaConf
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys, os
from evaluation import Evaluation
import pickle
from utils.logger import Logger
from time import sleep
import random
import torch
from torch import optim
from numba import cuda
from datetime import datetime
from nuscenes.eval.common.loaders import load_gt
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import *
from pyquaternion import Quaternion
from utils.camera import coord_2d_to_3d, sensor_coord_to_real_coord
from nuscenes.eval.common.utils import angle_diff, quaternion_yaw

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.multiprocessing.set_sharing_strategy('file_system')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/kiennt/KienNT/research/data/nuScenes/', verbose=False)
    data = pickle.load(open('/home/kiennt/KienNT/research/data/nuScenes/pickle/mini/val.pkl', 'rb'))
    result = {"meta":{"use_camera":True},"results":{}}
    eval_set = 'mini_val'
    gt_boxes = load_gt(nusc, eval_set, DetectionBox, verbose=False)
    sample_tokens = set(gt_boxes.sample_tokens)
    output_dir = '/home/kiennt/KienNT/research/test_result'
    result_path = os.path.join(output_dir, 'result_tmp.json')
    meta = pickle.load(open('/home/kiennt/KienNT/research/data/nuScenes/pickle/mini/meta.pkl', 'rb'))
    plot_examples = 20
    conf_th=0.05
    # print(meta)
    # exit()
    cfg_path = '/home/kiennt/KienNT/research/Detectron3D/config/detection_cvpr_2019.json'
    for sample_token in sample_tokens:
        boxes = [item for item in data if item['sample_token']==sample_token]
        
        test_boxes = []
        for box in boxes:
            for ann in box['annotations']:
                if meta['category_map'][ann['category'].split('.')[-1]]=='void':
                    continue
                # r = ann['rotation']
                rr = Quaternion(axis=[0, 0, 1], angle=ann['yaw_angle_rad'])
                
                box_real = sensor_coord_to_real_coord(
                            ann['xyz_in_sensor_coor'], ann['box_size'], rr, box['calibration_matrix'], rotate_yaw_only=True
                        )

                test_box = {
                    'sample_token': box['sample_token'],
                    # 'translation': ann['xyz_in_meter'],
                    'translation': box_real.center,
                    'size': ann['box_size'],
                    'rotation': box_real.orientation.elements,
                    'velocity': ann['velocity'][:2],
                    'detection_name': meta['category_map'][ann['category'].split('.')[-1]],
                    'detection_score': 1,
                    'attribute_name': ann['attribute'][0] if len(ann['attribute'])>0 else '',
                }
                test_boxes.append(test_box)
        # if sample_token=='3e8750f331d7499e9b5123e9eb70f2e2':
        #     print(len(test_boxes))
        #     exit()
        result['results'][sample_token] = test_boxes
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    out_file = open(result_path, "w")
    json.dump(result, out_file, cls=NpEncoder)
    out_file.close()
    
    with open(cfg_path, 'r') as _f:
        cfg_ = DetectionConfig.deserialize(json.load(_f))
    nusc_eval = DetectionEval(nusc, config=cfg_, result_path=result_path, eval_set=eval_set, output_dir=output_dir, verbose=False)
    
    if plot_examples > 0:
        # Select a random but fixed subset to plot.
        random.seed(42)
        sample_tokens = list(nusc_eval.sample_tokens)
        random.shuffle(sample_tokens)
        sample_tokens = sample_tokens[:plot_examples]

        # Visualize samples.
        example_dir = os.path.join(output_dir, 'examples')
        os.makedirs(example_dir, exist_ok=True)
        for sample_token in sample_tokens:
            visualize_sample(nusc,
                                sample_token,
                                nusc_eval.gt_boxes if nusc_eval.eval_set != 'test' else EvalBoxes(),
                                nusc_eval.pred_boxes,
                                conf_th=conf_th,
                                eval_range=max(nusc_eval.cfg.class_range.values()),
                                savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))
            
    metrics, metric_data_list = nusc_eval.evaluate()
    metrics_summary = metrics.serialize()
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)