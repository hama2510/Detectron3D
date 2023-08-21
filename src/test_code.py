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
        calibration_matrix = samples['calibration_matrix']
        for task_id in range(0, len(tasks)):
            model = tasks[task_id]['model']
            pred = model(imgs)
            loss, loss_log = criterion(targets, pred)
            # print(loss_log)
            t = targets['16']['category'][0].detach().cpu().numpy()
            idxs = np.argwhere(t==1)
            for idx in idxs:
                print(idx)
                print(pred['p4']['category'][0].shape)
                print(pred['p4']['category'][0][idx[2],idx[0],idx[1]])
                print(np.max(pred['p4']['category'][0].detach().cpu().numpy()[:,idx[0],idx[1]]))
            # print(sample_token[0])
            # img = imgs[0].detach().cpu().numpy()
            # category_map = targets['16']['category'][0].detach().cpu().numpy()
            # offset_map = targets['16']['offset'][0].detach().cpu().numpy()
            # cls_score = np.max(category_map, axis=2)
            # indices = np.argwhere(cls_score > 0)
            # for idx in indices:
            #     # sc = cls_score[idx[0], idx[1]]
            #     # y = int(idx[0] + offset_map[idx[0], idx[1], 1]) * 16
            #     # x = int(idx[1] + offset_map[idx[0], idx[1], 0]) * 16
            #     y = int(idx[0] * 16 + offset_map[idx[0], idx[1], 1])
            #     x = int(idx[1] * 16 + offset_map[idx[0], idx[1], 0])
            #     print(x,y)
            #     cv.circle(img, (x,y), 1, (0, 0, 255), 1)
            # cv.imwrite('./test.png', img)

            break
        break

