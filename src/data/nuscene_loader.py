from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import cv2 as cv
import os, sys
from pyquaternion import Quaternion
# from scipy.spatial.transform import Rotation as quaternion_transformer
import numpy as np
from functools import partial
from multiprocessing import Pool
import pickle
import argparse
# from sklearn.model_selection import train_test_split
sys.path.append('..')
from utils.camera import *
from tqdm import tqdm

class NuScenesLoader:
    def __init__(self, dataset_name, dataroot, num_worker=1, verbose=False):
        self.nusc = NuScenes(version=dataset_name, dataroot=dataroot, verbose=verbose)
        self.num_worker = num_worker

    def read_sensor(self, sample_data_token):
        sd_record = self.nusc.get('sample_data', sample_data_token)
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
#         return dict({
#             'sensor_R_quaternion': np.asarray(cs_record['rotation']),
#             'sensor_R': np.asarray(quaternion_transformer.from_quat(cs_record['rotation']).as_matrix()),
#             'sensor_t': np.asarray(cs_record['translation']),
#             'ego_R_quaternion': np.asarray(pose_record['rotation']),
#             'ego_R': np.asarray(quaternion_transformer.from_quat(pose_record['rotation']).as_matrix()),
#             'ego_t': np.asarray(pose_record['translation']),
#             'camera_intrinsic': np.asarray(cs_record['camera_intrinsic'])
#         })
        return dict({
            'sensor_R_quaternion': Quaternion(cs_record['rotation']).elements,
            'sensor_R': Quaternion(cs_record['rotation']).rotation_matrix,
            'sensor_t': np.asarray(cs_record['translation']),
            'ego_R_quaternion': Quaternion(pose_record['rotation']).elements,
            'ego_R': Quaternion(pose_record['rotation']).rotation_matrix,
            'ego_t': np.asarray(pose_record['translation']),
            'camera_intrinsic': np.asarray(cs_record['camera_intrinsic'])
        })

    def read_attribute(self, attr_tokens):
        attrs = []
        for attr_token in attr_tokens:
            attrs.append(self.nusc.get('attribute', attr_token)['name'])
        return attrs

    def read_visibility(self, visibility_token):
        return int(visibility_token)

    def read_sample(self, sample):
        data = dict()
        cams = [key for key in sample['data'].keys() if 'CAM' in key]
        for ann_token in sample['anns']:
            sample_data_token = None
            for cam in cams:
    #             Note that the boxes are transformed into the current sensor's coordinate frame.
                # sample_data_token = sample['data'][cam]
                data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample['data'][cam],
                                                                          selected_anntokens=[ann_token])
                if len(boxes) > 0:
                    sample_data_token = sample['data'][cam]
                    # data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token, selected_anntokens=[ann_token])
                    ann_metadata =  self.nusc.get('sample_annotation', ann_token)
                    assert len(ann_metadata['attribute_tokens'])<=1

                    if not data_path in data.keys():
                        data[data_path]={'anns':[], 'calibration_matrix':self.read_sensor(sample_data_token)}
                    for box in boxes:
        #                 print(box.orientation)
        #                 print(box.orientation.radians)
                        ann= dict({
                            'category': box.name,
                            'xyz_in_meter': ann_metadata['translation'],
                            'box_size': ann_metadata['size'],
                            'xyz_in_sensor_coor': box.center,
                            # using calibrated coord
                            'box_2d': box_3d_to_2d(box, data[data_path]['calibration_matrix']),
                            'rotation_angle_degree': box.orientation.degrees,
                            'rotation_angle_rad': box.orientation.radians,
                            'velocity': self.nusc.box_velocity(ann_token),
                            'attribute': self.read_attribute(ann_metadata['attribute_tokens']),
                            'visibility': self.read_visibility(ann_metadata['visibility_token'])
                        })
                        data[data_path]['anns'].append(ann)
        data = [{'scene':sample['scene_token'], 'sample_token': sample['token'], 'image':os.path.abspath(key), 'calibration_matrix':data[key]['calibration_matrix'], 
                 'annotations':data[key]['anns'], } for key in data.keys()]
        return data

    def get_all_scenes(self, ):
        return self.nusc.scene
    
    def get_samples_from_scene(self, scene):
        raw_data = []
        first_sample_token = scene['first_sample_token']
        curr_sample = self.nusc.get('sample', first_sample_token)
        while(curr_sample['next']!=''):
            raw_data.append(curr_sample)
            curr_sample = self.nusc.get('sample', curr_sample['next'])

        # pool = Pool(self.num_worker)
        # data = pool.map(self.read_sample, raw_data)
        # pool.close()
        # pool.join()
        merge_list = []
        for item in raw_data:
            merge_list.extend(self.read_sample(item))
        # for item in data:
        #     merge_list.extend(item)
        return merge_list
    
    def get_all_categories(self):
        return [item['name'] for item in self.nusc.category]

    def get_all_attributes(self):
        attributes = [item['name'] for item in self.nusc.attribute]
        attributes.append('void')
        return attributes

CATEGORY_MAP = {
            'animal': 'void',
            'debris': 'void',
            'pushable_pullable': 'void',
            'bicycle_rack': 'void',
            'ambulance': 'void',
            'police': 'void' ,
            'barrier': 'barrier' ,
            'bicycle': 'bicycle' ,
            'bendy': 'bus' ,
            'rigid': 'bus' ,
            'car': 'car' ,
            'construction': 'construction_vehicle',
            'motorcycle' :'motorcycle' ,
            'adult' :'pedestrian' ,
            'child': 'pedestrian' ,
            'construction_worker': 'pedestrian' ,
            'police_officer': 'pedestrian' ,
            'personal_mobility': 'void' ,
            'stroller': 'void' ,
            'wheelchair': 'void' ,
            'trafficcone': 'traffic_cone' ,
            'trailer': 'trailer' ,
            'truck': 'truck'
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, help='datset name')
    parser.add_argument('--dataroot', type=str, help='datset root')
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--out', type=str, help='out file folder')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    nusc = NuScenesLoader(dataset_name=args.dataset, dataroot=args.dataroot, num_worker=args.num_worker, verbose=True)
    splits = create_splits_scenes()
    train_data = []
    val_data = []
    scenes = nusc.get_all_scenes()
    if 'mini' in args.dataset:
        train_key='mini_train'
        val_key = 'mini_val'
    else:
        train_key='train'
        val_key = 'val'
        
    train_scenes = [item for item in scenes if item['name'] in splits[train_key]]
    if args.dataset=='v1.1-mini':
        train_scenes = [item for item in train_scences if item!='scene-0553']
    val_scenes = [item for item in scenes if item['name'] in splits[val_key]]
    
    for scene in tqdm(train_scenes):
        # print('Hanlding scene %s'%scene['name'])
        train_data.extend(nusc.get_samples_from_scene(scene))

    for scene in tqdm(val_scenes):
        # print('Hanlding scene %s'%scene['name'])
        val_data.extend(nusc.get_samples_from_scene(scene))
    categories = [CATEGORY_MAP[key] for key in CATEGORY_MAP.keys() if CATEGORY_MAP[key]!='void']
    attributes = nusc.get_all_attributes()

    meta = {'categories': categories,
            'category_map': CATEGORY_MAP, 
            'attributes': attributes}
    pickle.dump(meta, open(os.path.join(args.out, 'meta.pkl'), 'wb'))
    pickle.dump(train_data, open(os.path.join(args.out, 'train.pkl'), 'wb'))
    pickle.dump(val_data, open(os.path.join(args.out, 'val.pkl'), 'wb'))