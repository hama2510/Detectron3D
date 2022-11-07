import pandas as pd
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.nuscene_dataset import NusceneDataset
from model.detector import TrainDetector
from criterion.losses import Criterion
import argparse
from omegaconf import OmegaConf
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys, os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def run_valid(dataloader_val, model):
    for step, (imgs, y) in enumerate(tqdm(dataloader_val, desc="Valid", leave=False)):
#         for model_id in range(0, len(models)):
#         model = models[model_id]['model']

        imgs = imgs.to(config.device)
        pred = model(imgs)
        y = y.detach().cpu().numpy()
        