import os
import sys
sys.path.append("/home/diya/Public/Image2Smiles/jy/Diffusion-Zoo")

from diffusion.engines import ddpm
from datasets.cifar10_dataset import CIFAR10Dataset

import yaml
import time
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", type=str, required=True)
    parser.add_argument("--training_contd", type=str, default="")
    args = parser.parse_args()
    
    with open(args.cfg_dir, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    ckpt_name = f"{args.cfg_dir.split('/')[-1].split('.')[0]}_{cfg['dataset']['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    cfg['train']['ckpt_dir'] = os.path.join(cfg['train']['ckpt_dir'], ckpt_name)
    
    if not os.path.exists(cfg['train']['ckpt_dir']):
        os.makedirs(cfg['train']['ckpt_dir'])    
    
    with open(os.path.join(cfg['train']['ckpt_dir'], "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    
    cfg['model']['model_params']['self_condition'] = cfg['self_condition']
    engine = ddpm.DDPMEngine(cfg)
    engine.to(cfg['device'])
    dset = CIFAR10Dataset(cfg['dataset'])
    # build_dataset
    
    if args.training_contd:
        state_dict = torch.load(args.training_contd)
        engine.load_state_dict(state_dict)
    engine.train(cfg['train'], dset, None)