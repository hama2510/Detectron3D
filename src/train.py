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

torch.manual_seed(42)

def init_loss_log():
    loss_log = {}
    for stride in [8, 16, 32, 64, 128]:
        loss_log[stride] = {
            'category_loss': [],
            'attribute_loss': [],
            'offset_loss': [],
            'depth_loss': [],
            'size_loss': [],
            'rotation_loss': [],
            'dir_loss': [],
            'centerness_loss': [],
        }
    loss_log['total'] = []
    return loss_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',type=str, 
                        help='path_to_config_file')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    dataset_train = NusceneDataset(config.data.train, config=config)
    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    dataset_val = NusceneDataset(config.data.val, config=config)
    dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    criterion = Criterion(device=config.device)

    models = []
    for model_id, item in enumerate(config.models):
        model_config = config.copy()
        model_config.model = model_config.models[model_id]
        model = TrainDetector(model_config)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        models.append({'model':model, 'optimizer':optimizer, 'config':model_config, 'y_true':[], 'y_pred':[], 'best_f1':0, 'loss':init_loss_log()})
    
        if not os.path.exists(models[model_id]['config'].model.save_dir):
            os.makedirs(models[model_id]['config'].model.save_dir)
        f = open(os.path.join(models[model_id]['config'].model.save_dir, 'log.csv'), 'w')
        loss_log = init_loss_log()
        f.write('epoch,')
        for stride in loss_log.keys():
            for key in loss_log[stride].keys():
                f.write('{}_stride_{}'.format(key, stride))
        f.write('loss,f1,precision,recall\n')
        f.close()

    for epoch in range(1, config.epochs+1):
        # train
        for model_id in range(0, len(models)):
            models[model_id]['loss'] = []
        for step, samples in enumerate(tqdm(dataloader_train, desc="Train", leave=False)):
            for model_id in range(0, len(models)):
                model = models[model_id]['model']
                optimizer = models[model_id]['optimizer']
                
                imgs = samples['img']
                targets = samples['target']

                imgs = imgs.to(config.device)
                pred = model(imgs)
                optimizer.zero_grad()
                loss, loss_log = criterion(targets, pred)
                loss.backward()
                optimizer.step()
#                 
                models[model_id]['loss']['total'] = loss.cpu().detach().numpy()
                for stride in loss_log.keys():
                    for key in loss_log[stride].keys():
                        models[model_id]['loss'][stride][key].append(loss_log[stride][key])

#         # valid
#         for step, (imgs, y) in enumerate(tqdm(dataloader_val, desc="Valid", leave=False)):
#             for model_id in range(0, len(models)):
#                 model = models[model_id]['model']

#                 imgs = imgs.to(config.device)
#                 y = y.detach().cpu().numpy()
#                 y = np.argmax(y, axis=1).tolist()
#                 models[model_id]['y_true'].extend(y)
#                 pred = model(imgs).detach().cpu().numpy()
#                 pred = np.argmax(pred, axis=1).tolist()
#                 models[model_id]['y_pred'].extend(pred)

#         for model_id in range(0, len(models)):
#             y_true = models[model_id]['y_true']
#             y_pred = models[model_id]['y_pred']
#             model_config = models[model_id]['config']

#             weight = dataset_val.get_weight(weight_type='val')
#             sample_weight = [weight[item] for item in y_true]
#             precision = precision_score(y_true, y_pred, sample_weight=sample_weight, average=None, zero_division=0)
#             precision = np.mean(precision)
#             recall = recall_score(y_true, y_pred, sample_weight=sample_weight, average=None)
#             recall = np.mean(recall)
#             f1 = 2.0*precision*recall/(precision+recall)
#             if f1>models[model_id]['best_f1']:
#                 models[model_id]['best_f1']=f1
                
#             if config.save_best:
#                 if f1>=models[model_id]['best_f1']:
#                     # torch.save(model.state_dict(), os.path.join(config.save_dir, 'model_{}.pth'.format(epoch)))
#                     torch.save(model.state_dict(), os.path.join(model_config.save_dir, 'best_model.pth'))
#             else:
#                 torch.save(model.state_dict(), os.path.join(model_config.save_dir, 'model_{}.pth'.format(epoch)))

            f = open(os.path.join(model_config.save_dir, 'log.csv'), 'a')
            for stride in models[model_id]['loss']:
                for key in models[model_id]['loss'][stride].keys():
                    f.write('{},'.format(np.mean(models[model_id]['loss'][stride][key])))
            f.write('{},{},{},{},{}\n'.format(epoch, np.mean(models[model_id]['loss']['total']), f1, precision, recall))
            f.close()
            print('epoch={},model={},loss={},f1={},precision={},recall={}'.format(epoch, model_config.model.model_name, np.mean(models[model_id]['loss']), np.round(f1, decimals=2), np.round(precision, decimals=2), np.round(recall, decimals=2)))
        