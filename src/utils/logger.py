from datetime import datetime
import pickle
import os, sys
import numpy as np

class Logger:
    
    def __init__(self):
        self.log_file = 'log_{}.pkl'.format(datetime.now().strftime("%Y%m%d_%H%M"))

    def create_log_file(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pickle.dump([], open(os.path.join(save_dir, self.log_file), 'wb'))
        
    def init_loss_log(self, ):
        loss_log = {'total':[], 'component':{}}
        for stride in [8, 16, 32, 64, 128]:
            loss_log['component'][stride] = {
                'category_loss': [],
                'attribute_loss': [],
                'offset_loss': [],
                'depth_loss': [],
                'size_loss': [],
                'rotation_loss': [],
                'dir_loss': [],
                'centerness_loss': [],
            }
        return loss_log
        
    def log(self, item, save_dir):
        log_item = {}
        if 'loss' in item.keys():
            log_item = {'loss':{'total':np.mean(item['loss']['total']), 'component':{}}}
            for stride in item['loss']['component'].keys():
                log_item['loss']['component'][stride] = {}
                for key in item['loss']['component'][stride].keys():
                    log_item['loss']['component'][stride][key] = np.mean(item['loss']['component'][stride][key])
        for key in item.keys():
            if key!='loss':
                log_item[key] = item[key]
        log = pickle.load(open(os.path.join(save_dir, self.log_file), 'rb'))
        log.append(log_item)
        pickle.dump(log, open(os.path.join(save_dir, self.log_file), 'wb'))
        