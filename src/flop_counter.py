from model.fcos3d_detector import FCOSDetector, FCOSTransformer
import argparse
from omegaconf import OmegaConf
import torch
from datetime import datetime
from flopth import flopth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',type=str, 
                        help='path_to_config_file')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    for model_id, item in enumerate(config.models):
        model_config = config.copy()
        model_config.model = model_config.models[model_id]
        model = FCOSDetector(model_config)
        print(model_config.model.model_name)
        flops, params = flopth(model, in_size=((3, 1600, 900)))
        print(flops, params)
        img = torch.rand((1, 3, 1600,900)).to(model_config.device)
        start = datetime.now()
        _ = model(img)
        print('Prediction time: ', datetime.now()-start)
        print('---------------------')