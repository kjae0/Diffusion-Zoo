from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np


class MetricCalculator:
    def __init__(self, max_pixel_value=1.0, 
                 inception_block_idx=2048, 
                 cache_fid_stats=True,
                 cache_dir="./fid_stats.pth",
                 device=None):
        self.max_pixel_value = max_pixel_value
        self.device = device
        
        # for FID
        self.cache_fid_stats = cache_fid_stats
        self.cache_dir = cache_dir
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx])
        self.fid_stats = {
            'mean': None,
            'cov': None
        }
        
        
    def mse(self, image_gen, image_real):
        image_gen = image_gen.float()
        image_real = image_real.float()
        
        # Compute Mean Squared Error (MSE)
        mse = F.mse_loss(image_gen, image_real)
        
        return mse
        
    def psnr(self, image_gen, image_real):
        # Ensure the inputs are float tensors
        image_gen = image_gen.float()
        image_real = image_real.float()
        
        # Compute Mean Squared Error (MSE)
        mse = F.mse_loss(image_gen, image_real)
        
        if mse == 0:
            return float('inf')  # PSNR is infinite if MSE is 0
        
        # Compute PSNR
        psnr = 10 * torch.log10((self.max_pixel_value ** 2) / mse)
        
        return psnr

    def ssim(self, image_gen, image_real):
        pass
            
    def _fid_iteration(self, images_loader, device, verbose=False):        
        if verbose:
            dataloader = tqdm(images_loader, total=len(images_loader), desc="Calculating FID", ncols=100)
        else:
            dataloader = images_loader
        
        self.inception_v3.to(device)
        
        outs = []
        for x in dataloader:
            x = x.to(device)
            out = self.inception_v3(x)[0]
            
            if out.shape[2] != 1 or out.shape[3] != 1:
                out = F.adaptive_avg_pool2d(out, output_size=(1, 1))
            
            out = out.squeeze(-1).squeeze(-1)
            outs.append(out.cpu())

        outs = torch.cat(outs, dim=0)
        mean, cov = torch.mean(outs, dim=0), np.cov(outs, rowvar=False)
        
        return mean, cov
    
    def _get_fid_stats(self, fid_stat_dir, image_real, device, verbose=False):
        if fid_stat_dir:
            self.fid_stats = torch.load(fid_stat_dir)
        else:
            mean, cov = self._fid_iteration(image_real, device, verbose=verbose)
            self.fid_stats['mean'] = mean
            self.fid_stats['cov'] = cov
            
            if self.cache_fid_stats:
                torch.save(self.fid_stats, self.cache_dir)
    
    @torch.no_grad()
    def fid(self, image_gen_loader, image_real_loader=None, fid_stat_dir=None, verbose=False):
        assert (self.fid_stats['mean'] is not None and self.fid_stats['cov'] is not None) or image_real_loader is not None or fid_stat_dir is not None, "FID statistics not loaded"
                
        if self.fid_stats['mean'] is None or self.fid_stats['cov'] is None:
            print("Loading / Calculating FID statistics...")
            self._get_fid_stats(fid_stat_dir, image_real_loader, self.device, verbose=verbose)
            
        mean_gen, cov_gen = self._fid_iteration(image_gen_loader, self.device, verbose=verbose)
        
        return calculate_frechet_distance(mean_gen, cov_gen, self.fid_stats['mean'], self.fid_stats['cov'])
            
            
        
