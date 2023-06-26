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
from lion_pytorch import Lion


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
    dataset_val = NusceneDataset(config.data.val, config=config, return_target=False)
    dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    criterion = Criterion(device=config.device)
    evaluation = Evaluation(config.data.dataset_name, config.data.image_root, config.data.val_config_path)
    logger = Logger()
    transformer = FCOSTransformer(config)

    models = []
    for model_id, item in enumerate(config.models):
        model_config = config.copy()
        model_config.model = model_config.models[model_id]
        model = FCOSDetector(model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
#         optimizer = Lion(model.parameters(), lr=config.lr)
        models.append({'model':model, 'optimizer':optimizer, 'config':model_config, 'pred':[], 'best_score':0, 'loss': logger.init_loss_log()})
        logger.create_log_file(model_config.model.save_dir)
    
    for epoch in range(1, config.epochs+1):
        # train
        print('Training ...')
#         for step, samples in enumerate(dataloader_train):
        for model_id in range(0, len(models)):
            models[model_id]['model'].train()
        with tqdm(dataloader_train, desc="Train") as tepoch:
            for step, samples in enumerate(tepoch):
                loss_str = ''
                imgs = samples['img']
                targets = samples['target']
                imgs = imgs.to(config.device)
                for model_id in range(0, len(models)):
                    model = models[model_id]['model']
                    optimizer = models[model_id]['optimizer']

                    pred = model(imgs)
                    optimizer.zero_grad()
                    loss, loss_log = criterion(targets, pred)
                    loss.backward()
                    optimizer.step()
                    
                    models[model_id]['loss']['total'].append(loss.cpu().detach().numpy())
                    for stride in loss_log.keys():
                        for key in loss_log[stride].keys():
                            models[model_id]['loss']['component'][int(stride)][key].append(loss_log[stride][key])

                    loss_str+='{:.4f},'.format(np.mean(models[model_id]['loss']['total']))
                    del pred
                loss_str = loss_str[:-1]
                tepoch.set_postfix(ep=epoch, loss=loss_str)
                sleep(0.1)
#                 break
                
#         # valid
        print('Validating ...')
#         for step, samples in enumerate(dataloader_val):
        for model_id in range(0, len(models)):
            models[model_id]['model'].eval()
        for step, samples in enumerate(tqdm(dataloader_val, desc="Valid", leave=False)):
            imgs = samples['img']
            imgs = imgs.to(config.device)
            sample_token = samples['sample_token']
            calibration_matrix = samples['calibration_matrix']
            for model_id in range(0, len(models)):
                model = models[model_id]['model']
                pred = model(imgs)
                for i in range(len(sample_token)):
                    calib_matrix = {}
                    for key in calibration_matrix.keys():
                        calib_matrix[key] = calibration_matrix[key][i].detach().cpu().numpy()
                    item = {'sample_token':sample_token[i], 'calibration_matrix':calib_matrix, 'pred':{}}
                    for key in pred.keys():
                        item['pred'][key]={}
                        for sub_key in pred[key].keys():
                            item['pred'][key][sub_key] = model.item_tensor_to_numpy(sub_key, pred[key][sub_key][i])
                    models[model_id]['pred'].append(item)
                del pred
        start = datetime.now()
        for model_id in range(0, len(models)):
#             model = models[model_id]['model']
            preds = transformer.transform_predicts(models[model_id]['pred'])
            
            del models[model_id]['pred']
            models[model_id]['pred'] = []
            if len(preds)>0:
                if models[model_id]['config'].data.dataset_name == 'v1.0-mini':
                    metrics_summary = evaluation.evaluate(preds, eval_set='mini_val', verbose=False)
                else:
                    metrics_summary = evaluation.evaluate(preds, verbose=False)
                nds = metrics_summary['nd_score']
            else:
                metrics_summary = {}
                nds = 0
            
            logger.log({'epoch': epoch, 'loss': models[model_id]['loss'], 'metrics_summary':metrics_summary}, models[model_id]['config'].model.save_dir)
            print('epoch={},model={},loss={},nds={:.2f}'.format(epoch, models[model_id]['config'].model.model_name, np.mean(models[model_id]['loss']['total']), nds))
            
            del metrics_summary
            del models[model_id]['loss']
            del preds
            
            models[model_id]['loss'] = logger.init_loss_log()
            
            if config.save_best:
                if nds>models[model_id]['best_score']:
                    model.save_model(os.path.join(models[model_id]['config'].model.save_dir, 'model_{}.pth'.format(epoch)))
            else:
                model.save_model(os.path.join(models[model_id]['config'].model.save_dir, 'model_{}.pth'.format(epoch)))
            if nds>models[model_id]['best_score']:
                models[model_id]['best_score'] = nds
        print(datetime.now()-start)
