from diffusion.models import build_model
from diffusion.utils import get_beta_scheduler, Scaler, IdentityScaler
from diffusion.utils import build_loss_fn, build_optimizer, build_scheduler
from diffusion.utils import save_checkpoints, get_learning_rate
from diffusion.utils import shape_matcher_1d, ema
from diffusion.utils import x0_to_noise, noise_to_x0, velocity_to_x0
from diffusion.metrics import calculate_mse, calculate_psnr
from diffusion.models import unet_test

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
        sampling_timesteps = None,
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        # min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556  # TODO
        # min_snr_gamma = 5,
        immiscible = False
    ):
        self.model, self.time_embedding = build_model(cfg['model'])
    
        if cfg['ema']:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_decay = cfg['ema_decay']
        else:
            self.ema_model = None
        self.ema = cfg['ema']
        
        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))
        
        self.loss_fn = build_loss_fn(cfg['loss'])
        self.optimizer = build_optimizer(cfg['optimizer'], self.model.parameters())
        self.scheduler = build_scheduler(cfg['scheduler'], self.optimizer)
        
        self.device = cfg['device']
        self.verbose = cfg['verbose']
        self.clip_grad_norm = cfg['clip_grad_norm']
        self.offset_noise_strength = cfg['offset_noise_strength']
        self.image_size = cfg['image_size']
        self.timesteps = cfg['timesteps']
        self.self_cond = cfg['self_condition']
        self.loss_weight = cfg['min_snr_loss_weight']
        self.input_range = cfg['input_range']
        self.prediction_type = cfg['prediction_type']
        
        if self.loss_weight:
            raise NotImplementedError('min_snr_loss_weight is not supported yet')
        
        if self.input_range[0] != 0 or self.input_range[1] != 1 or self.input_range[0] > self.input_range[1] != 1:
            self.normalizer = Scaler(self.input_range)
        else:
            self.normalizer = IdentityScaler()
        
        betas = get_beta_scheduler(cfg['beta_scheduler'])(self.timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.)
        
        # predefine those for reduce repetitive computation
        self.one_minus_alphas_bar = 1. - alphas_bar
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
        self.p_sampling_coef_x0 = betas * torch.sqrt(alphas_bar_prev) / (1. - alphas_bar)
        self.p_sampling_coef_xt = torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar)
        self.betas_tilda = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
        self.log_betas_tilda = torch.log(self.betas_tilda.clamp(min =1e-20))
    
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else self.timesteps
        
        assert self.sampling_timesteps <= self.timesteps, 'sampling timesteps cannot be greater than total timesteps'

    def train(self, cfg, train_dataset, test_dataset, verbose=True):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                       batch_size=cfg['batch_size'], 
                                                       shuffle=True, 
                                                       drop_last=True,
                                                       num_workers=cfg['num_workers'])
        
        epochs = range(cfg['num_epochs'])
                    
        for epoch in epochs:
            train_loss, elapsed_time = self.train_one_epoch(train_dataloader, verbose=False, log_interval=cfg['log_interval'])
            
            print(f"{epoch+1} / {cfg['num_epochs']} Train Loss", end=" ")
            print(f"Train loss - {train_loss:6f}", end=" ")
            print(f"Time: {elapsed_time:2f} Lr: {get_learning_rate(self.optimizer)}")
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # evaluate
            if (epoch+1) % cfg['eval_interval'] == 0: 
                out, _ = self.p_sample([25, 3, 32, 32])
                save_images_grid(os.path.join(cfg['ckpt_dir'], f"{epoch+1}_images.png"), out)
                
            #     metric_dict = {'MSE': calculate_mse, 'PSNR': calculate_psnr}
            #     perf, preds, gts, elapsed_time = self.evaluate(cfg, test_dataset, metric_dict, hwf=(H, W, focal))

            #     print(f"\n{epoch+1} / {cfg['train']['num_epochs']} Eval Results")
            #     for k, v in perf.items():
            #         print(f"{k}: {v}")
            #     print(f"Time: {elapsed_time:2f}\n")                
            
            # TODO tensorboard logging...
            # if self.writer is not None:
                # self.writer.add_scalar("Train Loss", train_loss, epoch)
                            
            if (epoch+1) % cfg['save_interval'] == 0:
                ckpt = {
                    'diffusion': self.model.state_dict(),
                    'ema_model': self.ema_model.state_dict() if self.ema else None,
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
        
    def evaluate(self, cfg, test_dataset):
        test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                      batch_size=cfg['batch_size'], 
                                                      shuffle=False, 
                                                      drop_last=False,
                                                      num_workers=cfg['num_workers'])
        
        for x0, cls in test_dataloader:
            x0 = x0.to(self.device)
            x0 = self.normalizer.normalize(x0)
            cls = cls.to(self.device)
            
            noise = torch.randn_like(x0).to(self.device)
            
            if self.offset_noise_strength > 0.:
                offset_noise = torch.randn(x0.shape[:2], device=self.device)
                noise += self.offset_noise_strength * offset_noise.unsqueeze(-1).unsqueeze(-1)
            
            xt = self.q_sample(x0, noise, time_steps).float()
            
            x_self_cond = None
            if self.self_cond and random() < 0.5:
                with torch.no_grad():
                    x_self_cond = self.model_predictions(xt, time_steps).pred_x_start
                    x_self_cond.detach_()
            
            time_emb = self.time_embedding(time_steps)
            pred = self.model(xt, time_emb, x_self_cond=x_self_cond)
            pred = self.postprocessing(pred, xt, time_steps)
            
            loss = self.loss_fn(pred['noise'], noise)
            
            # TODO
            # conve rt target following prediction type
            loss = self.loss_fn(pred['noise'], noise)
            
            return loss.item()
        
    def run_network(self, x, time_step, self_cond=None, use_ema=False):
        """
            Run unet network. predict mean
        """
        # TODO: cond
        if self_cond is not None:
            self_cond = self_cond.float()
            
        time_emb = self.time_embedding(time_step)
        
        if use_ema:
            pred = self.ema_model(x.float(), time_emb, x_self_cond=self_cond)
        else:
            pred = self.model(x.float(), time_emb, x_self_cond=self_cond)
        # pred = self.model(x.float(), time_step)
        
        return pred
    
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
        # following ... ?
        # https://arxiv.org/pdf/2202.00512
        elif self.prediction_type == 'velocity':
            ret['x0'] = velocity_to_x0(pred, xt, self.sqrt_alphas_bar[time_steps], self.one_minus_alphas_bar[time_steps])
            ret['noise'] = x0_to_noise(ret['x0'], xt, self.sqrt_alphas_bar[time_steps], self.one_minus_alphas_bar[time_steps])
            
        # 3. predict x0
        # following DDIM?
        elif self.prediction_type == 'x0':
            ret['x0'] = pred
            ret['noise'] = x0_to_noise(pred, xt, self.sqrt_alphas_bar[time_steps], self.one_minus_alphas_bar[time_steps])
        
        if clip_x0:
            ret['x0'] = ret['x0'].clamp(self.input_range[0], self.input_range[1])
            
        return ret
    
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
            x0 = self.normalizer.normalize(x0)
            # cls = cls.to(self.device) # available only class guidance is used
            time_steps = time_steps.to(self.device)
            
            # B x 3 x H x W
            noise = torch.randn_like(x0).to(self.device)
            
            # TODO offset noise strength
            if self.offset_noise_strength > 0.:
                offset_noise = torch.randn(x0.shape[:2], device = self.device) # B x C
                noise += self.offset_noise_strength * offset_noise.unsqueeze(-1).unsqueeze(-1)
                
            xt = self.q_sample(x0, noise, time_steps).float()
            
            # if doing self-conditioning, 50% of the time, predict x_start from current set of times
            # and condition with unet with that
            # this technique will slow down training by 25%, but seems to lower FID significantly
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
            elif self.prediction_type == 'velocity':
                pass
            elif self.prediction_type == 'x0':
                pass
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
                # break
            
        return total_loss / n_iter, time.time() - s
    
    def ddim_sample(self,):
        pass
    
    def q_sample(self, x0, noise, time_steps):
        # TODO immiscrible
        
        # B x C x H x W
        noised = shape_matcher_1d(self.sqrt_alphas_bar[time_steps], x0.shape) * x0 + \
            shape_matcher_1d(self.sqrt_one_minus_alphas_bar[time_steps], noise.shape) * noise
            
        return noised
    
    @torch.no_grad()
    def p_sample(self, sample_shape=None, return_trajectory=False, verbose=True):
        self.model.eval()
        self.time_embedding.eval()

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

        return self.normalizer.unnormalize(xt.cpu()), ret
    
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
    
    def load_state_dict(self, state_dict):
        ckpt = state_dict['checkpoint']
        self.model.load_state_dict(ckpt['diffusion'])
        self.ema_model.load_state_dict(ckpt['ema_model'])
        self.time_embedding.load_state_dict(ckpt['time_embedding'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        
        print(f"Loaded state dict from epoch {state_dict['epoch']}")