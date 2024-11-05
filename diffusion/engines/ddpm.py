from diffusion.models import build_model
from diffusion.utils import get_beta_scheduler, Scaler, IdentityScaler
from diffusion.utils import save_checkpoints, get_learning_rate
from diffusion.utils import shape_matcher_1d, ema
from diffusion.utils import get_model_state_dict
from diffusion.metrics import MetricCalculator
from datasets import build_dataset

from diffusion.utils import (
    x0_to_noise, 
    noise_to_x0, 
    velocity_to_x0)
from diffusion.utils import (
    build_loss_fn, 
    build_optimizer, 
    build_scheduler, 
)

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


class DDPMEngine:
    def __init__(
        self,
        cfg,
        writer=None
    ):
        self.model, self.time_embedding = build_model(cfg['model'])
        self.writer = writer
        
        if cfg['ema']:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_decay = cfg['ema_decay']
        else:
            self.ema_model = None
        self.ema = cfg['ema']
        
        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('[INFO] Model params: %.2f M' % (model_size / 1024 / 1024))
        
        self.loss_fn = build_loss_fn(cfg['loss'])
        self.optimizer = build_optimizer(cfg['optimizer'], self.model.parameters())
        self.scheduler = build_scheduler(cfg['scheduler'], self.optimizer)
        self.dataset = build_dataset(cfg['dataset'])
        
        self.device = cfg['device']
        self.verbose = cfg['verbose']
        self.clip_grad_norm = cfg['clip_grad_norm']
        self.offset_noise_strength = cfg['offset_noise_strength']
        self.image_size = cfg['image_size']
        self.timesteps = cfg['timesteps']
        self.self_cond = cfg['model']['model_params']['self_condition']
        self.loss_weight = cfg['min_snr_loss_weight']
        self.input_range = cfg['input_range']
        self.prediction_type = cfg['prediction_type']
        self.ddim_sampling_eta = cfg['ddim_sampling_eta']
        
        if self.loss_weight:
            raise NotImplementedError('min_snr_loss_weight is not supported yet')
        
        if self.input_range[0] != 0 or self.input_range[1] != 1 or self.input_range[0] > self.input_range[1] != 1:
            self.scaler = Scaler(self.input_range)
        else:
            self.scaler = IdentityScaler()
        
        betas = get_beta_scheduler(cfg['beta_scheduler'])(self.timesteps)
        alphas = 1. - betas
        
        # predefine those for reduce repetitive computation
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.)
        self.one_minus_alphas_bar = 1. - self.alphas_bar
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)
        self.p_sampling_coef_x0 = betas * torch.sqrt(self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.p_sampling_coef_xt = torch.sqrt(alphas) * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.betas_tilda = betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.log_betas_tilda = torch.log(self.betas_tilda.clamp(min =1e-20))
    
        self.sampling_timesteps = cfg['sampling_timesteps'] if cfg['sampling_timesteps'] is not None else self.timesteps
        assert self.sampling_timesteps <= self.timesteps, 'sampling timesteps cannot be greater than total timesteps'

        self.metric_calculator = MetricCalculator(device=self.device, dataparallel=cfg['model']['dataparallel'], **cfg['metrics'])

    def run_network(self, x, time_step, self_cond=None, use_ema=False):
        if self_cond is not None:
            self_cond = self_cond.float()
            
        time_emb = self.time_embedding(time_step)
        
        if use_ema:
            pred = self.ema_model(x.float(), time_emb, x_self_cond=self_cond)
        else:
            pred = self.model(x.float(), time_emb, x_self_cond=self_cond)
        
        return pred

    def train(self, cfg):
        train_dataset = self.dataset
        test_dataset = self.dataset
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                       batch_size=cfg['batch_size'], 
                                                       shuffle=True, 
                                                       drop_last=True,
                                                       num_workers=cfg['num_workers'])
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=cfg['batch_size'] * 4,    # need less memory
                                                        shuffle=False,
                                                        drop_last=False,
                                                        num_workers=cfg['num_workers'])
        
        epochs = range(cfg['num_epochs'])            
        for epoch in epochs:
            # train one epoch
            train_loss, elapsed_time = self.train_one_epoch(train_dataloader, verbose=False, log_interval=cfg['log_interval'])
            
            print(f"[INFO] {epoch+1} / {cfg['num_epochs']} Train Loss", end=" ")
            print(f"Train loss - {train_loss:6f}", end=" ")
            print(f"Time: {elapsed_time:2f} Lr: {get_learning_rate(self.optimizer)}")
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # evaluate
            if (epoch+1) % cfg['vis_interval'] == 0: 
                out, _ = self.p_sample(sample_shape=[25] + self.image_size)
                save_images_grid(os.path.join(cfg['ckpt_dir'], f"{epoch+1}_images.png"), out)
                print(f"[INFO] Saved images at {cfg['ckpt_dir']}")
            
            if (epoch+1) % cfg['eval_interval'] == 0: 
                n_samples = cfg['n_eval_samples']
                fid_stats_dir = cfg['fid_stats_dir'] if 'fid_stats_dir' in cfg else None
                
                perf, elapsed_time = self.evaluate(n_samples, cfg['batch_size'], test_dataloader, fid_stats_dir, verbose=True)

                print(f"\n{epoch+1} / {cfg['num_epochs']} Eval Results")
                for k, v in perf.items():
                    print(f"{k}: {v}")
                print(f"Time: {elapsed_time:2f}\n")                
            
            # TODO tensorboard logging...
            if self.writer is not None:
                # add image
                self.writer.add_image('train/image', out, epoch)
                
                # add eval result
                for k, v in perf.items():
                    self.writer.add_scalar(f'eval/{k}', v, epoch)
                           
            # save checkpoints 
            if (epoch+1) % cfg['save_interval'] == 0:
                ckpt = {
                    'diffusion': get_model_state_dict(self.model),
                    'ema_model': get_model_state_dict(self.ema_model) if self.ema else None,
                    'time_embedding': self.time_embedding.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'random_states': {
                        'torch': torch.get_rng_state(),
                        'numpy': np.random.get_state(),
                        'python': random.getstate()
                    }
                }
                
                if not os.path.exists(cfg['ckpt_dir']):
                    os.makedirs(cfg['ckpt_dir'], exist_ok=True)
                    
                save_checkpoints(ckpt_path=os.path.join(cfg['ckpt_dir'], f"ckpt_{epoch+1}.pt"),
                                 ckpt=ckpt,
                                 epoch=epoch+1,
                                 train_loss=train_loss,
                                 val_loss=None)

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
            x0 = self.scaler.normalize(x0)
            # cls = cls.to(self.device) # available only class guidance is used
            time_steps = time_steps.to(self.device)
            
            # B x 3 x H x W
            noise = torch.randn_like(x0).to(self.device)
            
            # TODO offset noise strength
            if self.offset_noise_strength > 0.:
                offset_noise = torch.randn(x0.shape[:2], device = self.device) # B x C
                noise += self.offset_noise_strength * offset_noise.unsqueeze(-1).unsqueeze(-1)
                
            xt = self.q_sample(x0, noise, time_steps).float()
            
            x_self_cond = None
            if self.self_cond and random.random() < 0.5:
                with torch.no_grad():
                    pred = self.run_network(xt, time_steps, x_self_cond, use_ema=False)
                    pred = self.postprocessing(pred, xt, time_steps)
                    x_self_cond = pred['x0'].float().detach()
            
            # B x 3 x H x W
            pred = self.run_network(xt, time_steps, x_self_cond, use_ema=False)
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
                print(f"[INFO] {iter+1} / {len(dataloader)} Loss: ", loss.item(), f"Time Elapsed: {(time.time() - si):2f}")
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
            
        for b in sample_iter:
            out, _ = self.ddim_sample(sample_shape=[batch_size] + self.image_size)
            sampled_images.append(out)
        sampled_images = torch.cat(sampled_images, dim=0)   # n_samples x 3 x H x W
        
        sample_dl = torch.chunk(sampled_images, batch_size, dim=0)
        fid_score = self.metric_calculator.fid(sample_dl, test_dl, fid_stats_dir, verbose=verbose)
        ret['FID'] = fid_score
        
        return ret, time.time() - s
    
    def postprocessing(self, pred, xt, time_steps, clip_x0=True):
        # 3 modes
        ret = {
            'x0': None,
            'noise': None
        }
         
        # 1. predict noise
        # following DDPM / predict noise (mean)
        # https://arxiv.org/abs/2006.11239
        if self.prediction_type == 'noise':
            ret['x0'] = noise_to_x0(pred, xt, self.sqrt_alphas_bar[time_steps], self.sqrt_one_minus_alphas_bar[time_steps])
            ret['noise'] = pred
        # 2. predict velocity
        elif self.prediction_type == 'velocity':
            raise NotImplementedError('No QA for velocity prediction')
        # 3. predict x0
        elif self.prediction_type == 'x0':
            raise NotImplementedError('No QA for x0 prediction')
        if clip_x0:
            ret['x0'] = ret['x0'].clamp(self.input_range[0], self.input_range[1])
            
        return ret
    
    def q_sample(self, x0, noise, time_steps):
        # B x C x H x W
        noised = shape_matcher_1d(self.sqrt_alphas_bar[time_steps], x0.shape) * x0 + \
            shape_matcher_1d(self.sqrt_one_minus_alphas_bar[time_steps], noise.shape) * noise
            
        return noised
    
    @torch.no_grad()
    def ddim_sample(self, sample_shape=None, return_trajectory=False, clip_x0=True, eta=None, verbose=False):
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

        if verbose:
            time_pairs = tqdm(time_pairs, ncols=100, desc='Sampling')
        else:
            time_pairs = time_pairs
            
        for time, time_next in time_pairs:                
            out = self.ddim_sample_iter(xt, time, time_next, self_cond, clip_x0, eta=eta)
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
    def ddim_sample_iter(self, xt, t, t_next, self_cond, clip_x0=True, eta=None):
        assert len(xt.shape) == 4, 'input shape must be B x C x H x W'
        
        eta = eta if eta is not None else self.ddim_sampling_eta
        
        ret = {'x_t-1': None,
               'x0': None}
        
        B = xt.shape[0]
        batch_t = torch.full((B,), t, device=self.device, dtype=torch.long)    # B
            
        pred = self.run_network(xt, batch_t, self_cond, use_ema=self.ema)
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
    def p_sample(self, sample_shape=None, return_trajectory=False, verbose=True):
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
        for t in time_steps:
            out = self._p_sample_iter(xt, t, self_cond)
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
    def _p_sample_iter(self, xt, t, self_cond, clip_x0=True):
        assert len(xt.shape) == 4, 'input shape must be B x C x H x W'
        
        ret = {'x_t-1': None,
               'x0': None}
        
        B = xt.shape[0]
        batch_t = torch.full((B,), t, device=self.device, dtype=torch.long)    # B
        noise = torch.randn_like(xt) if t > 0 else 0.
        
        pred = self.run_network(xt, batch_t, self_cond, use_ema=self.ema)
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
        
    def to(self, device):
        self.model.to(device)
        self.time_embedding.to(device)
        self.device = device
        self.one_minus_alphas_bar = self.one_minus_alphas_bar.to(device)
        self.sqrt_alphas_bar = self.sqrt_alphas_bar.to(device)
        self.sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.to(device)
        self.p_sampling_coef_x0 = self.p_sampling_coef_x0.to(device)
        self.p_sampling_coef_xt = self.p_sampling_coef_xt.to(device)
        self.betas_tilda = self.betas_tilda.to(device)
        self.log_betas_tilda = self.log_betas_tilda.to(device)
        
        if self.ema:
            self.ema_model.to(device)
        
        return self
    
    def load_state_dict(self, state_dict, dataparallel=False):
        ckpt = state_dict['checkpoint']
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        
        if dataparallel:
            print("[INFO] Loading state dict for DataParallel model")
            self.model.module.load_state_dict(ckpt['diffusion'])
            self.time_embedding.module.load_state_dict(ckpt['time_embedding'])
            if self.ema:
                self.ema_model.module.load_state_dict(ckpt['ema_model'])
        else:
            self.model.load_state_dict(ckpt['diffusion'])
            self.time_embedding.load_state_dict(ckpt['time_embedding'])
            if self.ema:
                self.ema_model.load_state_dict(ckpt['ema_model'])
        
        print(f"[INFO] Loaded state dict from epoch {state_dict['epoch']} successfully.")
