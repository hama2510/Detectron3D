from model.detector.detector_factory import get_detector
import argparse
from omegaconf import OmegaConf
import torch
from datetime import datetime
from flopth import flopth
from thop import profile
import numpy as np
import pandas as pd
import os, sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, help="path_to_config_file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    df_data = []
    for model_id, item in enumerate(config.models):
        model_config = config.copy()
        model_config.model = model_config.models[model_id]
        model = get_detector(model_config.model.detector_name)(model_config)
        model = model.to(model_config.device)
        model.eval()
        print(model_config.model.exp)
        # flops, params = flopth(model, in_size=((3, 450, 800)))
        # flops, params = flopth(model, in_size=((3, 900, 1600)))
        # print(flops, params)
        h, w = model_config.model.input_shape
        input = torch.randn(1, 3, h, w).to(model_config.device)
        flops, params = profile(model, inputs=(input,))
        flops = np.round(flops / 1000000000, 2)
        params = np.round(params / 1000000, 2)

        output = model(input)
        memory_usage = torch.cuda.max_memory_allocated() / 1024**3
        # start = datetime.now()
        # _ = model(input)
        # print('Prediction time: ', datetime.now()-start)
        df_data.append(
            {
                "name": model_config.model.exp,
                "head": model_config.model.head_name,
                "fpn": model_config.model.fpn,
                "backbone": model_config.model.backbone_name,
                "input_size": (w, h),
                "flops": f"{flops}G",
                "params": f"{params}M",
                'memory_usage': memory_usage
            }
        )
        torch.cuda.reset_max_memory_allocated()
        print("---------------------")
    df_data = pd.DataFrame(df_data)
    os.makedirs(os.path.dirname(config.out_path), exist_ok=True)
    df_data.to_csv(config.out_path, index=False)
    print(df_data)
