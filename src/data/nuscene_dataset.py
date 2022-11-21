import sys, os
import torch
# import pandas as pd
# from skimage import io, transform
import pickle
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
sys.path.append('..')
from utils.camera import *

class NusceneDataset(Dataset):
    
    def __init__(self, data_file, config):
        self.image_root = config.data.image_root
        if not self.image_root.endswith('/'):
            self.image_root+='/'
        self.data = pickle.load(open(data_file, 'rb'))
        self.meta_data = pickle.load(open(config.data.meta_data, 'rb'))
        self.visibility_thres = config.visibility_thres
        self.stride_list = [8, 16, 32, 64, 128]
        self.m_list = [0, 64, 128, 256, 512, np.inf]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        item = self.data[idx]
#         img = cv.imread(self.image_root+item['image'])
        img = cv.imread(item['image'])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        shape = [img.shape[0], img.shape[1]]

        img = transforms.Compose([transforms.ToTensor()])(img.copy())
        sample = {'sample_token':item['sample_token'], 'calibration_matrix':item['calibration_matrix'], 'img':img, 'target':{}}
        for stride in self.stride_list:
            sample['target']['{}'.format(stride)] = self.gen_target(item['annotations'], shape, stride)
        return sample

    def rotation_angle_to_pi_and_bin(self, rotation_angle):
        rad, bin = rotation_angle/np.pi, np.max([0, int(np.sign(rotation_angle))])
        dir_cls = np.zeros(2)
        dir_cls[bin] = 1
        return rad, dir_cls

    def rotation_angle_to_pi_and_minus_pi(self, rotation_angle):
        return rotation_angle/2.0/np.pi

    def gen_category_onehot(self, category):
        category = self.meta_data['category_map'][category.split('.')[-1]]
        if category=='void':
            return None
        else:
            onehot = np.zeros(len(self.meta_data['categories']))
            onehot[self.meta_data['categories'].index(category)] = 1
            return onehot

    def gen_attribute_onehot(self, attribute):
        onehot = np.zeros(len(self.meta_data['attributes']))
        if len(attribute)==0:
            onehot[self.meta_data['attributes'].index('void')]
        else:
            onehot[self.meta_data['attributes'].index(attribute[0])] = 1
        return onehot

    def gen_velocity(self, velocity):
        return [velocity[0], velocity[1]]

    def check_box_and_feature_map_level(self, point, box, stride):
        (x1, y1), (x2, y2) = xywh_to_xyxy(box)
        l = point[0]-x1
        t = point[1]-y1
        r = x2-point[0]
        b = y2-point[1]
        m = np.max([l,t,b,r])
        if m<self.m_list[self.stride_list.index(stride)-1] or m>self.m_list[self.stride_list.index(stride)]:
            return False
        else:
            return True

    def centerness(self, point, box, alpha=2.5):
        return np.exp(-alpha*((point[0]-box[0][0])**2+(point[1]-box[0][1])**2))

    def offset(self, point, box, stride):
        return [box[0][0]-point[0]*stride, box[0][1]-point[1]*stride]

    def gen_target(self, anns, shape, stride):
        shape =  [int(np.ceil(shape[0]/stride)), int(np.ceil(shape[1]/stride))]
        category_target = np.zeros((shape[0], shape[1], len(self.meta_data['categories'])))
        attribute_target = np.zeros((shape[0], shape[1], len(self.meta_data['attributes'])))
        centerness_target = np.zeros((shape[0], shape[1], 1))
        offset_target = np.zeros((shape[0], shape[1], 2))
        depth_target = np.zeros((shape[0], shape[1], 1))
        size_target = np.zeros((shape[0], shape[1], 3))
        rotation_target = np.zeros((shape[0], shape[1], 1))
        dir_target = np.zeros((shape[0], shape[1], 2))
        velocity_target = np.zeros((shape[0], shape[1], 2))
        for x in range(shape[0]):
            for y in range(shape[1]):
                boxes = []
                for ann in anns:
                    if is_inside([x*stride, y*stride], ann['box_2d']) and ann['visibility']>self.visibility_thres and self.check_box_and_feature_map_level([x*stride, y*stride], ann['box_2d'], stride):
                        boxes.append(ann)
                if len(boxes)>0:
                    # foreground location
                    boxes.sort(key=lambda item: distance_to_center([x*stride, y*stride], item['box_2d']))
                    box = boxes[0]
                    box_2d = np.asarray(box['box_2d'], dtype=object)//stride
                    rad, dir_cls = self.rotation_angle_to_pi_and_bin(box['rotation_angle_rad'])

                    category_onehot = self.gen_category_onehot(box['category'])
                    if category_onehot is None:
                        # skip void objects
                        continue
                    category_target[x,y,:] = category_onehot
                    attribute_target[x,y,:] = self.gen_attribute_onehot(box['attribute'])
                    centerness_target[x,y,:] = self.centerness([x,y], box_2d)
                    offset_target[x,y,:] = self.offset([x,y], box['box_2d'], stride)
                    depth_target[x,y,:] = box['xyz_in_sensor_coor'][2]
                    size_target[x,y,:] = box['box_size']
                    rotation_target[x,y,:] = rad
                    dir_target[x,y,:] = dir_cls
                    velocity_target[x,y,:] = self.gen_velocity(box['velocity'])

        return {'category': torch.FloatTensor(category_target), 'attribute': torch.FloatTensor(attribute_target), 
                'centerness':torch.FloatTensor(centerness_target),  'offset': torch.FloatTensor(offset_target), 
                'depth': torch.FloatTensor(depth_target),  'size': torch.FloatTensor(size_target), 
                'rotation': torch.FloatTensor(rotation_target), 'dir': torch.FloatTensor(dir_target), 'velocity': torch.FloatTensor(velocity_target)
               }