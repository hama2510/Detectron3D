import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import sys, os
from evaluation import Evaluation
from utils.logger import Logger
from time import sleep
import random
import torch
from datetime import datetime
from lion_pytorch import Lion
from model.detector.detector_factory import get_detector
from model.transform.transform_factory import get_transform
from criterion.criterion_factory import get_criterion
from data.dataset_factory import get_dataset


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.multiprocessing.set_sharing_strategy("file_system")


class RunTask:
    def __init__(self, conf):
        self.conf = conf
        self.model = self.init_model()
        self.model = self.model.to(conf.device)
        self.optimizer = self.init_optimizer()
        self.criterion = self.init_criterion()
        self.transformer = self.init_transformer()

    def init_model(self):
        model = get_detector(self.conf.model.detector_name)(self.conf)
        return model

    def init_optimizer(self):
        if self.conf.train.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.train.lr)
        elif self.conf.train.optimizer == "Lion":
            optimizer = Lion(self.model.parameters(), lr=self.conf.train.lr)
        return optimizer

    def init_criterion(self):
        criterion = get_criterion(self.conf.train.loss)(self.conf.device)
        # criterion = criterion.to(self.conf.device)
        return criterion

    def init_transformer(self):
        return get_transform(self.conf.model.transform)(self.conf)

    def train(self, imgs, targets):
        imgs = imgs.to(self.conf.device)
        preds = self.model(imgs)
        self.optimizer.zero_grad()
        loss, loss_log = self.criterion(targets, preds)
        loss.backward()
        self.optimizer.step()

        del preds
        del imgs
        return loss, loss_log

    def valid(self, imgs, targets):
        imgs = imgs.to(self.conf.device)
        preds = self.model(imgs)
        loss, loss_log = self.criterion(targets, preds)

        del imgs
        return preds, loss, loss_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, help="path_to_config_file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    dataset = get_dataset(config.data.data_loader)

    dataset_train = dataset(config.data.train, config=config)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
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
                "best_nds": 0,
                "best_mAP": 0,
                "loss": logger.init_loss_log(),
                "val_loss": logger.init_loss_log(),
            }
        )
        logger.create_log_file(model_config.model.save_dir)

    for epoch in range(0, config.train.epochs + 1):
        if epoch>0:
        # train
            print("Training ...")
    #                 for step, samples in enumerate(dataloader_train):
            for task_id in range(0, len(tasks)):
                tasks[task_id].model.train()
            with tqdm(dataloader_train, desc="Train") as tepoch:
                for step, samples in enumerate(tepoch):
                    loss_str = ""
                    imgs = samples["img"]
                    targets = samples["target"]
                    for task_id in range(0, len(tasks)):
                        task = tasks[task_id]
                        loss, loss_log = task.train(imgs, targets)

                        logs[task_id]["loss"]["total"].append(loss.cpu().detach().numpy())
                        for stride in loss_log.keys():
                            for key in loss_log[stride].keys():
                                logs[task_id]["loss"]["component"][int(stride)][key].append(
                                    loss_log[stride][key]
                                )

                        loss_str += "{:.4f},".format(
                            np.mean(logs[task_id]["loss"]["total"])
                        )
                        del loss
                        del loss_log
                        del task
                    loss_str = loss_str[:-1]
                    tepoch.set_postfix(ep=epoch, loss=loss_str)
                    sleep(0.1)
                    del targets
                    del imgs

        if epoch>0 or (epoch==0 and 'load_eval' in config.keys() and config.load_eval):
        #         # valid
            print("Validating ...")
            #         for step, samples in enumerate(dataloader_val):
            for task_id in range(0, len(tasks)):
                tasks[task_id].model.eval()
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
                preds = task.transformer.transform_predicts(logs[id]["pred"])

                del logs[id]["pred"]
                logs[id]["pred"] = []
                if len(preds) > 0:
                    if task.conf.data.dataset_name == "v1.0-mini":
                        eval_set="mini_"
                    else:
                        eval_set = ""
                    if 'val.pkl' in task.conf.data.val:
                        eval_set+='val'
                    else:
                        eval_set+='train'
                    metrics_summary = evaluation.evaluate(preds, verbose=False, eval_set = eval_set)
                    nds = metrics_summary["nd_score"]
                    mAP = metrics_summary["mean_ap"]
                else:
                    metrics_summary = {}
                    nds = 0

                logger.log(
                    {
                        "epoch": epoch,
                        "loss": logs[id]["loss"],
                        "val_loss": logs[id]["val_loss"],
                        "metrics_summary": metrics_summary,
                    },
                    task.conf.model.save_dir,
                )
                print(
                    "epoch={},model={},loss={},nds={:.2f}".format(
                        epoch,
                        task.conf.model.exp,
                        np.mean(logs[id]["loss"]["total"]),
                        nds,
                    )
                )

                del metrics_summary
                del preds
                del logs[id]["loss"]
                del logs[id]["val_loss"]
                logs[id]["loss"] = logger.init_loss_log()
                logs[id]["val_loss"] = logger.init_loss_log()

                if task.conf.train.save_best:
                    if nds > logs[id]["best_nds"] or mAP>logs[id]["best_mAP"]:
                        task.model.save_model(
                            os.path.join(
                                task.conf.model.save_dir,
                                "model_{}.pth".format(epoch),
                            )
                        )
                else:
                    task.model.save_model(
                        os.path.join(
                            task.conf.model.save_dir,
                            "model_{}.pth".format(epoch),
                        )
                    )
                if nds > logs[id]["best_nds"]:
                    logs[id]["best_nds"] = nds
                if mAP > logs[id]["best_mAP"]:
                    logs[id]["best_mAP"] = mAP
                del task
            print(datetime.now() - start)
