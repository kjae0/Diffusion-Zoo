from diffusion.models import build_model
from diffusion.utils import get_beta_scheduler, Scaler, IdentityScaler
from diffusion.utils import build_loss_fn, build_optimizer, build_scheduler
from diffusion.utils import save_checkpoints, get_learning_rate
from diffusion.utils import shape_matcher_1d, ema
from diffusion.utils import x0_to_noise, noise_to_x0, velocity_to_x0
from diffusion.utils import get_model_state_dict
from diffusion.metrics import MetricCalculator
from diffusion.engines import ddpm

from src.utils import save_images_grid

from tqdm import tqdm

import os
import copy
import time
import random
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CFGEngine(ddpm.DDPMEngine):
    def __init__(
        self,
        cfg,
        writer=None
    ):
        if cfg['model']['model_params']['class_condition'] == False:
            print('[INFO] class_condition is False, setting to True')
            cfg['model']['model_params']['class_condition'] = True
        
        self.num_classes = cfg['model']['model_params']['num_classes']
        self.guidance_scale = cfg['guidance_scale']
        super().__init__(cfg)
    
    def run_network(self, x, y, time_step, 
                    class_mask_prob=0.1, class_cond_scale=None,
                    self_cond=None, use_ema=False):
        # TODO: cond
        if self_cond is not None:
            self_cond = self_cond.float()
            
        if use_ema:
            model = self.ema_model
        else:
            model = self.model
            
        time_emb = self.time_embedding(time_step)
        if class_cond_scale is not None:
            # For sampling (inference, not training)
            class_cond_mask = torch.zeros(x.size(0)) < 1.    # all True
            class_cond_mask = class_cond_mask.to(x.device)
            pred_class_cond = model(x.float(), time_emb, class_cond=y, class_cond_mask=class_cond_mask, x_self_cond=self_cond)
            
            class_cond_mask_null = torch.ones(x.size(0)) < 0.    # all False
            class_cond_mask_null = class_cond_mask_null.to(x.device)
            pred_null_cond = model(x.float(), time_emb, class_cond=y, class_cond_mask=class_cond_mask_null, x_self_cond=self_cond)
            
            pred = pred_class_cond + class_cond_scale * (pred_class_cond - pred_null_cond)
                
        else:
            class_cond_mask = torch.rand(x.size(0)) > class_mask_prob
            class_cond_mask = class_cond_mask.to(x.device)
            pred = model(x.float(), time_emb, class_cond=y, class_cond_mask=class_cond_mask, x_self_cond=self_cond)
        
        return pred
    
    def train_one_epoch(self, dataloader, verbose=True, log_interval=10):
        if verbose:
            dataloader = tqdm(dataloader, ncols=100, desc='Training...')
        
        total_loss = 0
        n_iter = 0
        s = time.time()
        si = time.time()
        for iter, (x0, cls) in enumerate(dataloader):
            # B
            time_steps = torch.randint(0, self.sampling_timesteps, (x0.shape[0],), device=self.device).long()
            
            # B x 3 x H x W
            x0 = x0.to(self.device)
            cls = cls.to(self.device)
            x0 = self.scaler.normalize(x0)
            # cls = cls.to(self.device) # available only class guidance is used
            time_steps = time_steps.to(self.device)
            
            # B x 3 x H x W
            noise = torch.randn_like(x0).to(self.device)
            
            if self.offset_noise_strength > 0.:
                offset_noise = torch.randn(x0.shape[:2], device = self.device) # B x C
                noise += self.offset_noise_strength * offset_noise.unsqueeze(-1).unsqueeze(-1)
                
            xt = self.q_sample(x0, noise, time_steps).float()
            
            x_self_cond = None
            if self.self_cond and random.random() < 0.5:
                with torch.no_grad():
                    pred = self.run_network(xt, cls, time_steps, self_cond=x_self_cond, use_ema=False)
                    pred = self.postprocessing(pred, xt, time_steps)
                    x_self_cond = pred['x0'].float().detach()
            
            # B x 3 x H x W
            pred = self.run_network(xt, cls, time_steps, self_cond=x_self_cond, use_ema=False)
            pred = self.postprocessing(pred, xt, time_steps)

            # TODO
            # convert target following prediction type
            if self.prediction_type == 'noise':
                loss = self.loss_fn(pred['noise'], noise)
            else:
                raise NotImplementedError(f"Prediction type {self.prediction_type} is not supported yet")
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            
            if self.ema:
                ema(self.model, self.ema_model, self.ema_decay)

            total_loss += loss.item()
            n_iter += 1
            
            if self.verbose and ((iter+1) % log_interval == 0):
                print(f"{iter+1} / {len(dataloader)} Loss: ", loss.item(), f"Time Elapsed: {(time.time() - si):2f}")
                si = time.time()

        return total_loss / n_iter, time.time() - s
    
    def evaluate(self, n_samples, batch_size, test_dl, fid_stats_dir=None, verbose=False):
        sampled_images = []
        ret = {
            'FID': None
        }
        
        s = time.time()
        if verbose:
            sample_iter = tqdm(range(0, n_samples, batch_size), ncols=100, desc='Sampling...')
        else:
            sample_iter = range(0, n_samples, batch_size)
            
        class_cond = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        for b in sample_iter:
            out, _ = self.ddim_sample(class_cond=class_cond, sample_shape=[batch_size] + self.image_size)
            sampled_images.append(out)
        sampled_images = torch.cat(sampled_images, dim=0)   # n_samples x 3 x H x W
        
        sample_dl = torch.chunk(sampled_images, batch_size, dim=0)
        fid_score = self.metric_calculator.fid(sample_dl, test_dl, fid_stats_dir, verbose=verbose)
        ret['FID'] = fid_score
        
        return ret, time.time() - s
    
    @torch.no_grad()
    def ddim_sample(self, class_cond=None, sample_shape=None, return_trajectory=False, clip_x0=True, eta=None):
        self.model.eval()
        self.time_embedding.eval()
        if self.ema:
            self.ema_model.eval()

        ret = []
        
        eta = eta if eta is not None else self.ddim_sampling_eta
        sample_shape = self.image_size if sample_shape is None else sample_shape
        
        xT = torch.randn(sample_shape).to(self.device)
        times = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        xt = xT
        self_cond = None
        
        if class_cond is None:
            class_cond = torch.arange(sample_shape[0], device=self.device, dtype=torch.long) % self.num_classes
        elif isinstance(class_cond, torch.Tensor):
            class_cond = class_cond.to(self.device)
        elif isinstance(class_cond, int):
            class_cond = torch.full((sample_shape[0],), class_cond, device=self.device, dtype=torch.long)
        elif isinstance(class_cond, list):
            class_cond = torch.tensor(class_cond, device=self.device, dtype=torch.long)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):                
            out = self.ddim_sample_iter(xt, class_cond, time, time_next, self_cond, clip_x0, eta=eta)
            xt = out['x_t-1']
            self_cond = out['x0']
            
            if return_trajectory:
                ret.append(xt.cpu())
                
        self.model.train()
        self.time_embedding.train()
        if self.ema:
            self.ema_model.train()

        return self.scaler.unnormalize(xt.cpu()), ret
    
    @torch.no_grad()
    def ddim_sample_iter(self, xt, class_cond, t, t_next, self_cond, clip_x0=True, eta=None):
        assert len(xt.shape) == 4, 'input shape must be B x C x H x W'
        
        eta = eta if eta is not None else self.ddim_sampling_eta
        
        ret = {'x_t-1': None,
               'x0': None}
        
        B = xt.shape[0]
        batch_t = torch.full((B,), t, device=self.device, dtype=torch.long)    # B
            
        pred = self.run_network(xt, class_cond, batch_t, self_cond=self_cond, use_ema=self.ema, class_cond_scale=self.guidance_scale)
        pred = self.postprocessing(pred, xt, batch_t, clip_x0)        

        if t_next < 0:
            return {
                'x0': pred['x0'],
                'x_t-1': pred['x0']
            }

        alpha = self.alphas_bar[t]
        alpha_next = self.alphas_bar[t_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(xt)

        ret['x_t-1'] = pred['x0'] * alpha_next.sqrt() + \
                c * pred['noise'] + \
                sigma * noise
        ret['x0'] = pred['x0']

        return ret
    
    @torch.no_grad()
    def p_sample(self, class_cond=None, sample_shape=None, return_trajectory=False, verbose=True):
        self.model.eval()
        self.time_embedding.eval()
        if self.ema:
            self.ema_model.eval()

        ret = []
        
        sample_shape = self.image_size if sample_shape is None else sample_shape

        xT = torch.randn(sample_shape).to(self.device)
        
        time_steps = reversed(range(self.sampling_timesteps))
        if verbose:
            time_steps = tqdm(time_steps, ncols=100, total=self.sampling_timesteps, desc='Sampling...') 
        
        self_cond = None
        xt = xT
        
        if class_cond is None:
            class_cond = torch.arange(sample_shape[0], device=self.device, dtype=torch.long) % self.num_classes
            
            print("[WARNING] DEBUGGING MODE!!! ALL SAMPLES WOULD BE TRUCKS")
            class_cond = torch.full((sample_shape[0],), 0, device=self.device, dtype=torch.long)
            
        elif isinstance(class_cond, torch.Tensor):
            class_cond = class_cond.to(self.device)
        elif isinstance(class_cond, int):
            class_cond = torch.full((sample_shape[0],), class_cond, device=self.device, dtype=torch.long)
        elif isinstance(class_cond, list):
            class_cond = torch.tensor(class_cond, device=self.device, dtype=torch.long)
            
        for t in time_steps:
            out = self._p_sample_iter(xt, class_cond, t, self_cond)
            xt = out['x_t-1']
            self_cond = out['x0']
            
            if return_trajectory:
                ret.append(xt.cpu())
                
        self.model.train()
        self.time_embedding.train()
        if self.ema:
            self.ema_model.train()

        return self.scaler.unnormalize(xt.cpu()), ret
    
    @torch.no_grad()
    def _p_sample_iter(self, xt, class_cond, t, self_cond, clip_x0=True):
        assert len(xt.shape) == 4, 'input shape must be B x C x H x W'
        
        ret = {'x_t-1': None,
               'x0': None}
        
        B = xt.shape[0]
        batch_t = torch.full((B,), t, device=self.device, dtype=torch.long)    # B
        noise = torch.randn_like(xt) if t > 0 else 0.
        
        pred = self.run_network(xt, class_cond, batch_t, self_cond=self_cond, class_cond_scale=self.guidance_scale, use_ema=self.ema)
        pred = self.postprocessing(pred, xt, batch_t, clip_x0)        

        mean = pred['x0'] * shape_matcher_1d(self.p_sampling_coef_x0[t], pred['x0'].shape) + \
            xt * shape_matcher_1d(self.p_sampling_coef_xt[t], xt.shape)
        var = self.betas_tilda[t]             # B
        var_log = self.log_betas_tilda[t]     # B
        
        # TODO check var clipping
        x_updated = mean + (0.5 * var_log).exp() * noise
        
        ret['x_t-1'] = x_updated
        ret['x0'] = pred['x0']
        
        return ret
    
