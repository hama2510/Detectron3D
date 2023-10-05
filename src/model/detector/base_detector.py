import torch
import torch.nn as nn
import numpy as np

from model.module.mobilenet_v2 import MobileNetv2
from model.module.resnet101 import ResNet101
from model.module.resnet101_deformable import ResNet101DCN
from model.module.efficientnet_v2 import EfficientNetV2S
import pickle
from collections import OrderedDict


class BaseDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.meta_data = pickle.load(open(config.data.meta_data, "rb"))
        self.init()

    def forward(self, x):
        return self.model(x)

    def init(self):
        self.model = self.create_model()
        if "load_model" in self.config.model.keys() and self.config.model.load_model:
            self.load_model(self.config.model.load_model)
            print("Loaded weight from {}".format(self.config.model.load_model))
        if "multi_gpu" in self.config.keys() and self.config.multi_gpu:
            if "gpus" in self.config:
                self.model = nn.DataParallel(self.model, device_ids=self.config.gpus)
            else:
                self.model = nn.DataParallel(self.model)
        if self.config.model["eval"]:
            self.model.eval()

    def create_head(self):
        raise NotImplementedError()

    def create_model(self):
        detector = self.create_head()
        if self.config.model.backbone_name == "mobilenet":
            model = detector(
                feature_extractor=MobileNetv2(self.config.device, pretrained=True),
                num_cate=len(self.meta_data["categories"]),
                num_attr=len(self.meta_data["attributes"]),
            )
        elif self.config.model.backbone_name == "resnet101":
            model = detector(
                feature_extractor=ResNet101(self.config.device, pretrained=True),
                num_cate=len(self.meta_data["categories"]),
                num_attr=len(self.meta_data["attributes"]),
            )
        elif self.config.model.backbone_name == "resnet101_dcn":
            model = detector(
                feature_extractor=ResNet101DCN(),
                num_cate=len(self.meta_data["categories"]),
                num_attr=len(self.meta_data["attributes"]),
            )
        elif self.config.model.backbone_name == "efficientnet_v2_s":
            model = detector(
                feature_extractor=EfficientNetV2S(self.config.device, pretrained=True),
                num_cate=len(self.meta_data["categories"]),
                num_attr=len(self.meta_data["attributes"]),
            )
        else:
            print("Not support backbone {}".format(self.config.model.backbone_name))
            exit()
        model.to(self.config["device"])
        return model

    def save_model(self, path):
        if self.config.multi_gpu:
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

    def item_tensor_to_numpy(self, key, item):
        raise NotImplementedError()

    def get_head_keys(self):
        raise NotImplementedError()

    def tensor_to_numpy(self, pred):
        output = {
            "sample_token": pred["sample_token"],
            "calibration_matrix": pred["calibration_matrix"],
            "pred": {},
        }
        for key in pred["pred"].keys():
            output["pred"][key] = {}
            head_keys = self.get_head_keys()
            for head_key in head_keys:
                map = pred["pred"][key][head_key].detach().cpu().numpy()
                map = np.moveaxis(map, 0, -1)
                output["pred"][key][head_key] = map
        return output
