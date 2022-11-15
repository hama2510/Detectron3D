from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import *
import json

class Evaluation:
    def __init__(self, dataset_name, dataroot, eval_config, verbose=False):
        self.nusc = NuScenes(version=dataset_name, dataroot=dataroot, verbose=verbose)
        self.cfg_path = eval_config
        self.output_dir = '/home/hotta/kiennt/Detectron3D/tmp/'
        self.result_path = '/home/hotta/kiennt/Detectron3D/tmp/result_tmp.json'
        
    def evaluate(self, preds, eval_set='val'):
        result = {'submission': {"meta": { "use_camera":   True}, "results": preds}}
        out_file_tmp = open(self.result_path, "w")
        json.dump(result, out_file, indent = 6)
        out_file.close()
        nusc_eval = DetectionEval(self.nusc, config=eval_config, result_path=self.result_path, eval_set=eval_set, output_dir=self.output_dir, verbose=verbose)
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)
        return metrics_summary