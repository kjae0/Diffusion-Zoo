from diffusion.nn.norms import RMSNorm

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn


class UpSamplingBlock(nn.Module):
    def __init__(self, in_dim, out_dim, factor=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor = factor, mode = 'nearest'),
            nn.Conv2d(in_dim, out_dim, 3, padding = 1)
        )
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.block[1].weight)
        nn.init.zeros_(self.block[1].bias)
    
    def forward(self, x):
        return self.block(x)


class DownSamplingBlock(nn.Module):
    def __init__(self, in_dim, out_dim, factor=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1, stride=factor),
        )
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.block[0].weight)
        nn.init.zeros_(self.block[0].bias)
        
    def forward(self, x):
        return self.block(x)


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0., padding=1, kernel_size=3, act_fn=nn.SiLU(), norm_fn=RMSNorm):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding = padding)
        self.norm = norm_fn(out_dim)
        self.act = act_fn
        self.dropout = nn.Dropout(dropout)
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        
        return self.dropout(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim=None, dropout = 0.):
        super().__init__()
        
        if time_emb_dim:
            # check
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_dim * 2)
            )
        else:
            self.mlp = None

        self.block1 = Conv2dBlock(in_dim, out_dim, dropout = dropout)
        self.block2 = Conv2dBlock(out_dim, out_dim)
        self.res_conv = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def _init_weights(self):
        self.block1._init_weights()
        self.block2._init_weights()
        
        if not isinstance(self.res_conv, nn.Identity):
            nn.init.kaiming_normal_(self.res_conv.weight)
            nn.init.zeros_(self.res_conv.bias)

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        
        if self.mlp:
            assert time_emb is not None, 'time embedding must be given'
            
            if class_emb is not None:
                time_emb = torch.cat((time_emb, class_emb), dim=-1)
            
            time_emb = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
            scale_shift = time_emb.chunk(2, dim = 1) 

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)
