import os
import sys
sys.path.append("/home/diya/Public/Image2Smiles/jy/Diffusion-Zoo")

from diffusion.engines import ddpm, classifier_free_guidance

import yaml
import time
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--training_contd", type=str, default="")
    args = parser.parse_args()
    
    # load config
    with open(args.cfg_dir, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # setting for ckpt
    ckpt_name = f"{args.cfg_dir.split('/')[-1].split('.')[0]}_{cfg['dataset']['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    cfg['train']['ckpt_dir'] = os.path.join(cfg['train']['ckpt_dir'], ckpt_name)
    
    if not os.path.exists(cfg['train']['ckpt_dir']):
        os.makedirs(cfg['train']['ckpt_dir'])    
    
    # for check config of ckpt
    with open(os.path.join(cfg['train']['ckpt_dir'], "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    
    # build engine
    if args.model == "ddpm":
        print("Building DDPM engine...")
        engine = ddpm.DDPMEngine(cfg)
    elif args.model == "cfg":
        print("Building Classifier Free Guidance engine...")
        engine = classifier_free_guidance.CFGEngine(cfg)
    engine.to(cfg['device'])
    
    # if you continue training
    if args.training_contd:
        state_dict = torch.load(args.training_contd)
        engine.load_state_dict(state_dict, dataparallel=cfg['model']['dataparallel'])
        
    # let's go!
    engine.train(cfg['train'])
    