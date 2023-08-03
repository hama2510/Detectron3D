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
from valid import Evaluation
import pickle
from utils.logger import Logger
from time import sleep
import random
import torch
from torch import optim
from numba import cuda
from datetime import datetime


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',type=str, 
                        help='path_to_config_file')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    dataset_train = NusceneDataset(config.data.train, config=config)
    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    dataset_val = NusceneDataset(config.data.val, config=config, return_target=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    criterion = Criterion(device=config.device)
    logger = Logger()
    transformer = FCOSTransformer(config)

    tasks = []
    for model_id, item in enumerate(config.models):
        model_config = config.copy()
        model_config.model = model_config.models[model_id]
        model = FCOSDetector(model_config)
        model.eval()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        tasks.append({'model':model, 'optimizer':optimizer, 'config':model_config, 'pred':[], 'best_score':0, 'loss': logger.init_loss_log()})
    
#     for step, samples in enumerate(dataloader_val):
    for step, samples in enumerate(tqdm(dataloader_val, desc="Valid", leave=False)):
        imgs = samples['img']
        imgs = imgs.to(config.device)
        sample_token = samples['sample_token']
        targets = samples['target']
        img_paths = samples['img_path']

        calibration_matrix = samples['calibration_matrix']
        for task_id in range(0, len(tasks)):
            model = tasks[task_id]['model']
            pred = model(imgs)

            for i in range(len(sample_token)):
                calib_matrix = {}
                for key in calibration_matrix.keys():
                    calib_matrix[key] = calibration_matrix[key][i].detach().cpu().numpy()
                item = {'sample_token':sample_token[i], 'calibration_matrix':calib_matrix, 'pred':{}, 'img_path':img_paths[i]}
                for key in pred.keys():
                    item['pred'][key]={}
                    for sub_key in pred[key].keys():
                        item['pred'][key][sub_key] = model.item_tensor_to_numpy(sub_key, pred[key][sub_key][i])
                tasks[task_id]['pred'].append(item)
            del pred
#         break
    start = datetime.now()
    for task_id in range(0, len(tasks)):
#             model = tasks[task_id]['model']
        preds = transformer.transform_predicts(tasks[task_id]['pred'])
        evaluation = Evaluation(config.data.dataset_name, config.data.image_root, config.data.val_config_path, output_dir=tasks[task_id]['config'].model.save_dir)
        if len(preds)>0:
            if tasks[task_id]['config'].data.dataset_name == 'v1.0-mini':
                metrics_summary = evaluation.evaluate(preds, eval_set='mini_val', verbose=False, clear=False, plot_examples=20,conf_th=config.det_thres)
            else:
                metrics_summary = evaluation.evaluate(preds, verbose=False,  clear=False, plot_examples=20,conf_th=config.det_thres)
            nds = metrics_summary['nd_score']
        else:
            metrics_summary = {}
            nds = 0
            
