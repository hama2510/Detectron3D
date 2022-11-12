import torch
import torch.nn as nn
import numpy as np
from .fcos3d import FCOS3D
from .mobilenet_v2 import MobileNetv2
from .resnet101 import ResNet101
from .resnet101_deformable import ResNet101DCN
import pickle

class TrainDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.meta_data = pickle.load(open(config.data.meta_data, 'rb'))
        self.model = self.create_model()
        self.load_model()

    def forward(self, x):
        return self.model(x)
    
    def create_model(self,):
        if self.config.model.model_name=='mobilenet':
            model = FCOS3D(feature_extractor=MobileNetv2(self.config.device, pretrained=True), num_cate=len(self.meta_data['categories']), num_attr=len(self.meta_data['attributes']))
        # elif 'efficientnet' in self.config.model.model_name:
        #     model = EfficientNet.from_pretrained(self.config.model.model_name, num_classes=self.config.model.num_class)
        elif self.config.model.model_name=='resnet101':
            model = FCOS3D(feature_extractor=ResNet101(self.config.device, pretrained=True), num_cate=len(self.meta_data['categories']), num_attr=len(self.meta_data['attributes']))
        elif self.config.model.model_name=='resnet101_dcn':
            model = FCOS3D(feature_extractor=ResNet101DCN(), num_cate=len(self.meta_data['categories']), num_attr=len(self.meta_data['attributes']))
        else:
            print('Not support model {}'.format(config.model.model_name))
            exit()
        if self.config['multi_gpu']:
            model = nn.DataParallel(model)
        model.to(self.config['device'])
        return model
     
    def load_model(self, ):
        if 'load_model' in self.config.model.keys() and self.config.model.load_model:
            print('Loaded weight from {}'.format(self.config.model.load_model))
            state_dict = torch.load(self.config.model.load_model)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[6:] # remove `model.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        if self.config.model['eval']:
            self.model.eval()
            
class Predictor:
    def __init__(self, config):
        self.config = config
        self.meta_data = pickle.load(open(config.data.meta_data, 'rb'))
        
    def transform_predict(self, pred):
        boxes = []
        for key in pred.keys():
            level = int(key[1:])
            stride = 2**level
            category_map = pred['category'].cpu().detach().numpy()
            attribute_map = pred['attribute'].cpu().detach().numpy()
            centerness_map = pred['centerness'].cpu().detach().numpy()
            offset_map = pred['offset'].cpu().detach().numpy()
            size_map = pred['size'].cpu().detach().numpy()
            rotation_map = pred['rotation'].cpu().detach().numpy()
            dir_map = pred['dir'].cpu().detach().numpy()
            velocity_map = pred['velo'].cpu().detach().numpy()
            
            cls_score = np.max(category_map, axis=1)
            pred_score = cls_score*centerness_map
