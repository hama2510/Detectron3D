from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import *

class Evaluation:
    def __init__(self, dataset_name, dataroot, eval_config, verbose=False):
        self.nusc = NuScenes(version=dataset_name, dataroot=dataroot, verbose=verbose)
        self.cfg_path = eval_config
        self.output_dir = '/home/hotta/kiennt/Detectron3D/tmp/'
        self.result_path = '/home/hotta/kiennt/Detectron3D/tmp/result_tmp.json'
        
    def evaluate(self, pred, eval_set='val'):
        nusc_eval = DetectionEval(self.nusc, config=eval_config, result_path=self.result_path, eval_set=eval_set, output_dir=self.output_dir, verbose=verbose)
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)
        return metrics_summary
    
      