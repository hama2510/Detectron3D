import torch
import torch.nn as nn
import numpy as np
from .fcos3d import FCOS3D
from .mobilenet_v2 import MobileNetv2
from .resnet101 import ResNet101
from .resnet101_deformable import ResNet101DCN
import pickle
from pyquaternion import Quaternion
sys.path.append('..')
from utils.nms import rotated_nms


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

    def transform_predict(self, pred, thres=0.05):
        boxes = []
        sample_token = pred['sample_token']
        calib_matrix = pred['calibration_matrix']
        for key in pred['pred'].keys():
            level = int(key[1:])
            stride = 2**level
            
            category_map = pred['pred'][key]['category']
            attribute_map = pred['pred'][key]['attribute']
            centerness_map = pred['pred'][key]['centerness']
            offset_map = pred['pred'][key]['offset']
            depth_map = pred['pred'][key]['depth']
            size_map = pred['pred'][key]['size']
            rotation_map = pred['pred'][key]['rotation']
            dir_map = pred['pred'][key]['dir']
            velocity_map = pred['pred'][key]['velo']
            
            cls_score = np.max(category_map, axis=0)
            pred_score = cls_score*centerness_map
            indices = np.argwhere(pred_score>thres)
            for idx in indices:
                sc = pred_score[idx[0], idx[1]]
                x, y = int(idx[0]+offset_map[0][idx[0]][idx[1]])*stride, int(idx[1]+offset_map[1][idx[0]][idx[1]])*stride
                depth = np.exp(depth_map[0][idx[0]][idx[1]])
                coord_3d = coord_2d_to_3d([x, y], depth, calib_matrix)
                size = np.exp(size_map[:][idx[0]][idx[1]])
                rotation = rotation_map[0][idx[0]][idx[1]]
                dir = dir_map[0][idx[0]][idx[1]]
                if dir<0.5:
                    rotation = -rotation
                velocity = velocity_map[0][idx[0]][idx[1]]
                category = self.meta_data['category'][np.argmax(category_map[0][idx[0]][idx[1]])]
                attribute = self.meta_data['attribute'][np.argmax(attribute_map[0][idx[0]][idx[1]])]
                
                boxes.append({
                    'sample_token': sample_token
                    'translation': coord_3d,
                    'size': size,
                    'rotation': Quaternion(axis=[1, 0, 0], angle=rotation),
                    'rotation_angle': rotation,
                    'velocity': velocity,
                    'detection_name': category,
                    'detection_score': sc,
                    'attribute_name': attribute,
                })
        keep_indices = rotated_nms(boxes, calib_matrix)
        boxes = [boxes[i] for i in keep_indices]
        return boxes
