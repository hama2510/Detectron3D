import torch
import torch.nn as nn
import numpy as np
from .fcos3d import FCOS3D
from .mobilenet_v2 import MobileNetv2
from .resnet101 import ResNet101
from .resnet101_deformable import ResNet101DCN
import pickle
from pyquaternion import Quaternion
import sys
sys.path.append('..')
from utils.nms import rotated_nms
from utils.camera import coord_2d_to_3d, sensor_coord_to_real_coord

class FCOSDetector(nn.Module):
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

    def transform_predict(self, pred, thres=0.0):
        boxes = []
        sample_token = pred['sample_token']
        calib_matrix = pred['calibration_matrix']
        for key in pred['pred'].keys():
#             stride = int(key)
            stride = 2**int(key[1:])
            
            category_map = pred['pred'][key]['category'].detach().cpu().numpy()
            attribute_map = nn.functional.softmax(pred['pred'][key]['attribute'], dim=1).detach().cpu().numpy()
            centerness_map = pred['pred'][key]['centerness'].detach().cpu().numpy()
            offset_map = pred['pred'][key]['offset'].detach().cpu().numpy()
            depth_map = pred['pred'][key]['depth'].detach().cpu().numpy()
            size_map = pred['pred'][key]['size'].detach().cpu().numpy()
            rotation_map = pred['pred'][key]['rotation'].detach().cpu().numpy()
            dir_map = nn.functional.softmax(pred['pred'][key]['dir'], dim=1).detach().cpu().numpy()
            velocity_map = pred['pred'][key]['velocity'].detach().cpu().numpy()
            
            category_map = np.moveaxis(category_map, 0, -1)
            attribute_map = np.moveaxis(attribute_map, 0, -1)
            centerness_map = np.moveaxis(centerness_map, 0, -1)
            offset_map = np.moveaxis(offset_map, 0, -1)
            depth_map = np.moveaxis(depth_map, 0, -1)
            size_map = np.moveaxis(size_map, 0, -1)
            rotation_map = np.moveaxis(rotation_map, 0, -1)
            dir_map = np.moveaxis(dir_map, 0, -1)
            velocity_map = np.moveaxis(velocity_map, 0, -1)
            
            cls_score = np.max(category_map, axis=2)
            pred_score = cls_score*centerness_map[:,:,0]
            indices = np.argwhere(pred_score>thres)
            indices = np.unique(indices, axis=0)
            for idx in indices:
                sc = pred_score[idx[0], idx[1]]
                x, y = int(idx[0]*stride+offset_map[idx[0], idx[1],0]), int(idx[1]*stride+offset_map[idx[0], idx[1],1])
                depth = np.exp(depth_map[idx[0]][idx[1],0])
#                 depth = depth_map[idx[0],idx[1],0]
                coord_3d = coord_2d_to_3d([x, y], depth, calib_matrix)
                size = size_map[idx[0],idx[1],:]
#                 rotation = rotation_map[idx[0],idx[1],0]*np.pi
                rotation = np.arcsin(rotation_map[idx[0],idx[1],0])
                dir = np.argmax(dir_map[idx[0],idx[1],:])
                if dir==0:
                    if rotation<0:
                        rotation-=np.pi/2
                    else:
                        rotation+=np.pi/2
                else:
                    if rotation<0:
                        rotation+=np.pi/2
                    else:
                        rotation-=np.pi/2
                rotation_q = Quaternion(axis=[0, 0, 1], angle=rotation)
                velocity = velocity_map[idx[0],idx[1],:]
                category = self.meta_data['categories'][np.argmax(category_map[idx[0],idx[1],:])]
                if category in ['barrier', 'traffic_cone'] :
                    attribute = ''
                else:
                    attribute = self.meta_data['attributes'][np.argmax(attribute_map[idx[0],idx[1],:])]
                    if attribute=='void':
                        attribute = ''
                        
                boxes.append({
                    'sample_token': sample_token,
                    'translation': sensor_coord_to_real_coord(coord_3d, size, rotation_q, calib_matrix),
                    'size': size,
                    'rotation': rotation_q.elements,
                    'rotation_angle': rotation,
                    'velocity': velocity,
                    'detection_name': category,
                    'detection_score': sc,
                    'attribute_name': attribute,
                })
        keep_indices = rotated_nms(boxes, calib_matrix)
        boxes = [boxes[i] for i in keep_indices]
        return boxes
    
    def transform_predicts(self, preds, thres=0.05):
        boxes = []
        for pred in preds:
            boxes.extend(self.transform_predict(pred, thres))
        return boxes
    