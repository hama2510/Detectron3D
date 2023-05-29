import pandas as pd
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.nuscene_dataset import NusceneDataset
from model.fcos3d_detector import FCOSDetector, FCOSTransformer
from model.centernet3d_detector import CenterNet3DDetector, CenterNet3DTransformer
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

def get_detector(model_config):
    if model_config.model.detector_name=='fcos3d':
        return FCOSDetector(model_config), FCOSTransformer(model_config)
    elif model_config.model.detector_name=='centernet3d':
        return CenterNet3DDetector(model_config), CenterNet3DTransformer(model_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',type=str, 
                        help='path_to_config_file')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    a = pickle.load(open('/home/kiennt/KienNT/research/data/nuScenes/pickle/mini/val.pkl', 'rb'))
    print(a)
    # models = []
    # for model_id, item in enumerate(config.models):
    #     model_config = config.copy()
    #     model_config.model = model_config.models[model_id]
    #     model, transformer = get_detector(model_config)
    #     # models.append({'model':model, 'transformer':transformer, 'optimizer':optimizer, 'config':model_config, 'pred':[], 'best_score':0, 'loss': logger.init_loss_log()})

    #     input = torch.rand((4, 3, 1600, 900))
    #     out = model(input)
    #     print(out['p4'].keys())
