import math
import torch
import torch.nn as nn

def remove_module_from_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def get_model_state_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def x0_to_noise(x0, xt, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod):
    return (xt - shape_matcher_1d(sqrt_alpha_cumprod, x0.shape) * x0) / \
        shape_matcher_1d(sqrt_one_minus_alpha_cumprod, x0.shape)

def noise_to_x0(noise, xt, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod):
    return (xt - shape_matcher_1d(sqrt_one_minus_alpha_cumprod, noise.shape) * noise) / \
        shape_matcher_1d(sqrt_alpha_cumprod, noise.shape)

def velocity_to_x0(velocity, xt, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod):
    return shape_matcher_1d(sqrt_alpha_cumprod, xt.shape) * xt - \
        shape_matcher_1d(sqrt_one_minus_alpha_cumprod, velocity.shape) * velocity

def shape_matcher_1d(x:torch.Tensor, shape):
    return x.view(-1, *((1,) * (len(shape)-1)))

def get_learning_rate(optimizer):
    # PyTorch optimizers can have multiple parameter groups, each with its own learning rate.
    # Here, we retrieve the learning rate of the first parameter group.
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def save_checkpoints(ckpt_path, ckpt, epoch, train_loss, val_loss=None):
    torch.save({
        'epoch': epoch,
        'checkpoint': ckpt,
        'train_loss': train_loss,
        'val_loss': val_loss
    }, ckpt_path)

def build_loss_fn(cfg):
    if cfg['name'] == 'l2':
        return nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss function {cfg['loss_fn']} not implemented")
    
def build_optimizer(cfg, params):
    if cfg['name'] == 'adam':
        return torch.optim.Adam(params, **cfg['optimizer_params'])
    else:
        raise NotImplementedError(f"Optimizer {cfg['optimizer']} not implemented")
    
def build_scheduler(cfg, optimizer):
    if cfg['name'] == 'none':
        return None
    elif cfg['name'] == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **cfg['scheduler_params'])
    elif cfg['name'] == 'multisteplr':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **cfg['scheduler_params'])
    else:
        raise NotImplementedError(f"Scheduler {cfg['name']} not implemented")
    
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def get_beta_scheduler(sche_name):
    if sche_name == 'linear':
        return linear_beta_schedule
    if sche_name == 'cosine':
        return cosine_beta_schedule
    if sche_name == 'sigmoid':
        return sigmoid_beta_schedule
    raise ValueError(f'unknown beta schedule {sche_name}')

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Scaler:
    def __init__(self, range):
        self.range = range
        
    def normalize(self, x: torch.Tensor):
        # x -> B x C x H x W range 0 to 1
        return x * (self.range[1] - self.range[0]) + self.range[0]
    
    def unnormalize(self, x: torch.Tensor):
        # x -> B x C x H x W range range[0] to range[1]
        return (x - self.range[0]) / (self.range[1] - self.range[0])
    
class IdentityScaler:
    def __init__(self):
        pass
    
    def normalize(self, x: torch.Tensor):
        return x
    
    def unnormalize(self, x: torch.Tensor):
        return x

