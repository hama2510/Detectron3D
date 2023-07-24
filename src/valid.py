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
    def __init__(self, dataset_name, dataroot, eval_config, verbose=False, output_dir='../tmp/'):
        self.nusc = NuScenes(version=dataset_name, dataroot=dataroot, verbose=verbose)
#         if 'mini' in dataset_name:
#             self.eval_set = 'mini_val'
        self.cfg_path = eval_config
        self.output_dir = output_dir
        self.result_path = os.path.join(self.output_dir, 'result_tmp.json')
        
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
    
    def clear(self, ):
         shutil.rmtree(self.output_dir) 
            
    def evaluate(self, preds, eval_set='val', verbose=False, max_box=100, clear=True, plot_examples=0,conf_th=0.05):
        gt_boxes = load_gt(self.nusc, eval_set, DetectionBox, verbose=verbose)
        sample_tokens = set(gt_boxes.sample_tokens)
        result = {"meta":{"use_camera":True},"results":{}}
        for sample_token in sample_tokens:
            boxes = [item for item in preds if item['sample_token']==sample_token]
            boxes.sort(key=lambda x: x["detection_score"])
            if len(boxes) > max_box:
                boxes = boxes[:max_box]
#             if len(boxes)==0:
#                 boxes = [self.dummy_box(sample_token)]
            result['results'][sample_token] = boxes
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        out_file = open(self.result_path, "w")
        json.dump(result, out_file, cls=NpEncoder)
        out_file.close()
        
        with open(self.cfg_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))
        nusc_eval = DetectionEval(self.nusc, config=cfg_, result_path=self.result_path, eval_set=eval_set, output_dir=self.output_dir, verbose=verbose)
        
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(nusc_eval.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            os.makedirs(example_dir, exist_ok=True)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 nusc_eval.gt_boxes if nusc_eval.eval_set != 'test' else EvalBoxes(),
                                 nusc_eval.pred_boxes,
                                 conf_th=conf_th,
                                 eval_range=max(nusc_eval.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))
                
        metrics, metric_data_list = nusc_eval.evaluate()
        metrics_summary = metrics.serialize()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        metrics_summary = json.load(open(os.path.join(self.output_dir, 'metrics_summary.json'), 'r'))
        if clear:
            self.clear() 
        return metrics_summary