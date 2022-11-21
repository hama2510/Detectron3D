from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import *
import json
import os
import shutil
from nuscenes.eval.common.loaders import load_gt
from nuscenes.eval.detection.data_classes import DetectionBox

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class Evaluation:
    def __init__(self, dataset_name, dataroot, eval_config, verbose=False):
        self.nusc = NuScenes(version=dataset_name, dataroot=dataroot, verbose=verbose)
#         if 'mini' in dataset_name:
#             self.eval_set = 'mini_val'
        self.cfg_path = eval_config
        self.output_dir = '/home/hotta/kiennt/Detectron3D/tmp/'
        self.result_path = '/home/hotta/kiennt/Detectron3D/tmp/result_tmp.json'
        
    def evaluate(self, preds, eval_set='val', verbose=True):
        gt_boxes = load_gt(self.nusc, eval_set, DetectionBox, verbose=verbose)
        sample_tokens = set(gt_boxes.sample_tokens)
        result = {"meta":{"use_camera":True},"results":{}}
        for sample_token in sample_tokens:
            result['results'][sample_token] = [item for item in preds if item['sample_token']==sample_token]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_file = open(self.result_path, "w")
        json.dump(result, out_file, cls=NpEncoder)
        out_file.close()
        with open(self.cfg_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))
        nusc_eval = DetectionEval(self.nusc, config=cfg_, result_path=self.result_path, eval_set=eval_set, output_dir=self.output_dir, verbose=verbose)
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)
#         shutil.rmtree(self.output_dir) 
        return metrics_summary