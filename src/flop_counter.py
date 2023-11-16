from model.detector.detector_factory import get_detector
import argparse
from omegaconf import OmegaConf
import torch
from datetime import datetime
from flopth import flopth
from thop import profile
import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',type=str, 
                        help='path_to_config_file')
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

        input = torch.randn(1, 3, 450, 800).to(model_config.device)
        flops, params = profile(model, inputs=(input, ))
        flops = np.round(flops/1000000000, 2)
        params = np.round(params/1000000, 2)
        # print(f'flops={flops}G, params={params}M')
        # input = torch.randn(1, 3, 450, 800).to(model_config.device)
        # start = datetime.now()
        # _ = model(input)
        # print('Prediction time: ', datetime.now()-start)
        df_data.append({'model':model_config.model.exp, 'flops':f'{flops}G', 'params':f'{params}M'})
        print('---------------------')
    df_data = pd.DataFrame(df_data)
    print(df_data)