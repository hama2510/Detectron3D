import pandas as pd
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.detector.detector_factory import get_detector
from model.transform.transform_factory import get_transform
from criterion.criterion_factory import get_criterion
from data.dataset_factory import get_dataset
import argparse
from omegaconf import OmegaConf
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys, os
from evaluation import Evaluation
import pickle
from utils.logger import Logger
import random
import torch
from torch import optim
from datetime import datetime
from train import RunTask


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, help="path_to_config_file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    dataset = get_dataset(config.data.data_loader)
    
    dataset_val = dataset(config.data.val, config=config, return_target=True)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    evaluation = Evaluation(
        config.data.dataset_name, config.data.image_root, config.data.val_config_path
    )
    logger = Logger()

    tasks = []
    logs = []
    for model_id, item in enumerate(config.models):
        model_config = config.copy()
        model_config.model = model_config.models[model_id]
        model_config.model.save_dir = os.path.join(model_config.model.save_dir, model_config.model.exp)
        task = RunTask(model_config)
        tasks.append(task)
        logs.append(
            {
                "pred": [],
                "best_score": 0,
                "loss": logger.init_loss_log(),
                "val_loss": logger.init_loss_log(),
            }
        )
#     for step, samples in enumerate(dataloader_val):
    for step, samples in enumerate(tqdm(dataloader_val, desc="Valid", leave=False)):
            imgs = samples["img"]
            imgs = imgs.to(config.device)
            sample_token = samples["sample_token"]
            calibration_matrix = samples["calibration_matrix"]
            targets = samples["target"]
            img_paths = samples["img_path"]

            for task_id in range(0, len(tasks)):
                task = tasks[task_id]
                preds, loss, loss_log = task.valid(imgs, targets)
                logs[task_id]["val_loss"]["total"].append(loss.cpu().detach().numpy())
                for stride in loss_log.keys():
                    for key in loss_log[stride].keys():
                        logs[task_id]["val_loss"]["component"][int(stride)][key].append(
                            loss_log[stride][key]
                        )
                for i in range(len(sample_token)):
                    calib_matrix = {}
                    for key in calibration_matrix.keys():
                        calib_matrix[key] = (
                            calibration_matrix[key][i].detach().cpu().numpy()
                        )
                    item = {
                        "sample_token": sample_token[i],
                        "calibration_matrix": calib_matrix,
                        "pred": {},
                        "img_path": img_paths[i],
                    }
                    for key in preds.keys():
                        item["pred"][key] = {}
                        for sub_key in preds[key].keys():
                            item["pred"][key][
                                sub_key
                            ] = task.model.item_tensor_to_numpy(
                                sub_key, preds[key][sub_key][i]
                            )
                    logs[task_id]["pred"].append(item)

                del preds
                del task
                del loss
                del loss_log
            del targets
            del imgs
    start = datetime.now()
    for id in range(0, len(tasks)):
        task = tasks[id]
        preds = task.transformer.transform_predicts(logs[id]["pred"], target=False)

        if len(preds) > 0:
            if task.conf.data.dataset_name == "v1.0-mini":
                eval_set="mini_"
            else:
                eval_set = ""
            if 'val.pkl' in task.conf.data.val:
                eval_set+='val'
            else:
                eval_set+='train'
            metrics_summary = evaluation.evaluate(
                preds,
                verbose=False,
                clear=False,
                eval_set = eval_set,
                output_dir=task.conf.model.save_dir,
                plot_examples=10
            )
            nds = metrics_summary["nd_score"]
        else:
            metrics_summary = {}
            nds = 0

            
