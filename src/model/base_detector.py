import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from .network.fcos3d import FCOS3D
from .network.mobilenet_v2 import MobileNetv2
from .network.resnet101 import ResNet101
from .network.resnet101_deformable import ResNet101DCN
import pickle
from pyquaternion import Quaternion
import sys, os
sys.path.append('..')
from utils.nms import rotated_nms
from utils.camera import coord_2d_to_3d, sensor_coord_to_real_coord
from datetime import datetime
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
from datetime import datetime

class BaseDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.meta_data = pickle.load(open(config.data.meta_data, 'rb'))
        self.init()

    def forward(self, x):
        return self.model(x)
    
    def init(self, ):
        self.model = self.create_model()
        if 'load_model' in self.config.model.keys() and self.config.model.load_model:
            self.load_model(self.config.model.load_model)
            print('Loaded weight from {}'.format(self.config.model.load_model))
        if 'multi_gpu'in self.config.keys() and self.config.multi_gpu:
            if 'gpus' in self.config:
#                 self.model = DistributedDataParallel(self.model, device_ids=self.config.gpus)
                self.model = nn.DataParallel(self.model, device_ids=self.config.gpus)
            else:
#                 self.model = DistributedDataParallel(self.model)
                self.model = nn.DataParallel(self.model)
        if self.config.model['eval']:
            self.model.eval()
    
    def create_model(self,):
        pass
    
    def save_model(self, path):
        new_state_dict = OrderedDict()
        if self.config.multi_gpu:
            torch.save(self.model.module.state_dict(), path)
#             for k, v in self.model.module.state_dict().items():
#                 new_state_dict[k] = v
        else:
            torch.save(self.model.state_dict(), path)
#             for k, v in self.model.state_dict().items():
#                 new_state_dict[k] = v
#         torch.save(new_state_dict, path)
     
    def load_model(self, path):
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        
    def item_tensor_to_numpy(self, key, item):
        if key=='category':
            item = torch.clamp(item, min=1e-4, max=1-1e-4)
        elif key=='attribute' or key=='dir':
            item = nn.functional.softmax(item, dim=1)
        item = item.detach().cpu().numpy()
        item = np.moveaxis(item, 0, -1)
        return item
        
    def tensor_to_numpy(self, pred):
        pass
        return output