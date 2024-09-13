import math
import torch
import torch.nn as nn

from einops import rearrange

# TODO check dimensional validity
# argurment -> 2 * dim?
class SinusoidalEmbedding(nn.Module):
    def __init__(self, in_dim, theta=10000):
        super().__init__()
        self.in_dim = in_dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.in_dim // 2
        
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb

class RandomOrLearnedSinusoidalEmbedding(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, in_dim, is_random = False):
        super().__init__()
        assert in_dim % 2 == 0, 'dimension must be divisible by 2'
        half_dim = in_dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        
        return fouriered
    
    
class TimeEmbedding(nn.Module):
    def __init__(self, in_dim,
                       hidden_dim_factor=4,
                       learned_sinusoidal_cond=False,
                       random_fourier_features=False,
                       learned_sinusoidal_dim=16,
                       sinusoidal_pos_emb_theta=10000,):
        super().__init__()
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        hidden_dim = in_dim * hidden_dim_factor

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalEmbedding(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalEmbedding(in_dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = in_dim
        
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )    
    
    def forward(self, x):
        return self.time_mlp(x)
    