import os
import sys
import argparse
import yaml
import torch

sys.path.append("/home/diya/Public/Image2Smiles/jy/Diffusion-Zoo")

from tqdm import tqdm
from diffusion.utils import remove_module_from_state_dict
from diffusion.engines import ddpm, classifier_free_guidance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dset", type=str, default="cifar10")
    args = parser.parse_args()
    
    with open(os.path.join(args.ckpt_dir, "config.yaml"), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    cfg['device'] = 'cuda'
    cfg['model']['dataparallel'] = True
    
    if args.model == "ddpm":
        print("[INFO] Building DDPM engine...")
        engine = ddpm.DDPMEngine(cfg)
    elif args.model == "cfg":
        print("[INFO] Building Classifier Free Guidance engine...")
        cfg['guidance_scale'] = 7.5
        engine = classifier_free_guidance.CFGEngine(cfg)
    else:
        raise ValueError("Invalid model name")
        
    engine.to(cfg['device'])

    ckpts = os.listdir(args.ckpt_dir)
    ckpts = [ckpt for ckpt in ckpts if ckpt.endswith('.pt')]
    ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    sd = torch.load(os.path.join(args.ckpt_dir, ckpts[-1]), map_location='cuda')
    print(f"[INFO] Loading checkpoint from {ckpts[-1]}")

    sd['checkpoint']['diffusion'] = remove_module_from_state_dict(sd['checkpoint']['diffusion'])
    sd['checkpoint']['ema_model'] = remove_module_from_state_dict(sd['checkpoint']['ema_model'])
    sd['checkpoint']['time_embedding'] = remove_module_from_state_dict(sd['checkpoint']['time_embedding'])

    engine.load_state_dict(sd, dataparallel=True)
    engine.sampling_timesteps = 100
    perf, elapsed_time = engine.evaluate(args.n_samples, args.batch_size, test_dl=None, 
                        fid_stats_dir=f"/home/diya/Public/Image2Smiles/jy/Diffusion-Zoo/fid_{args.dset}_stats.pth",
                        verbose=True)
    
    print("Eval Results")
    for k, v in perf.items():
        print(f"{k}: {v}")
    print(f"Time: {elapsed_time:2f}\n") 
    
