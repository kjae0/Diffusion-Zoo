import sys
sys.path.append("/home/diya/Public/Image2Smiles/jy/Diffusion-Zoo")

from diffusion.nn.blocks import ResnetBlock, DownSamplingBlock, UpSamplingBlock
from diffusion.nn.attention import Attention2d, LinearAttention

from einops.layers.torch import Rearrange

import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, in_dim, time_emb_dim,
        init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3,
        self_condition=False, learned_variance=False,
        dropout=0., attn_dim_head=32, attn_heads=4, full_attn=None, flash_attn=False
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = init_dim if in_dim else in_dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: in_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = full_attn if isinstance(full_attn, tuple) else (full_attn,) * num_stages 
        attn_heads = attn_heads if isinstance(attn_heads, tuple) else (attn_heads,) * num_stages 
        attn_dim_head = attn_dim_head if isinstance(attn_dim_head, tuple) else (attn_dim_head,) * num_stages 
        
        assert len(full_attn) == len(dim_mults) == len(attn_heads) == len(attn_dim_head), 'length of full_attn, attn_heads, attn_dim_head should be equal to length of dim_mults'

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            layer = nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_emb_dim, dropout = dropout),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_emb_dim, dropout = dropout)
            ])
            
            if layer_full_attn:
                layer.append(Attention2d(dim_in, 
                                       attn_dim = layer_attn_dim_head, 
                                       heads = layer_attn_heads,
                                       flash = flash_attn))
            else:
                layer.append(LinearAttention(dim_in, 
                                              attn_dim = layer_attn_dim_head, 
                                              heads = layer_attn_heads))
            
            if ind >= (num_resolutions - 1):
                layer.append(nn.Conv2d(dim_in, dim_out, 3, padding=1))
            else:
                layer.append(DownSamplingBlock(dim_in, dim_out))
                

            self.downs.append(layer)            

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim, dropout=dropout)
        self.mid_attn = Attention2d(mid_dim, heads=attn_heads[-1], attn_dim=attn_dim_head[-1], flash=flash_attn)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim, dropout=dropout)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            layer = nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_emb_dim, dropout=dropout),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_emb_dim, dropout=dropout)
            ])
            
            if layer_full_attn:
                layer.append(Attention2d(dim_out, 
                                       attn_dim = layer_attn_dim_head, 
                                       heads = layer_attn_heads,
                                       flash = flash_attn))
            else:
                layer.append(LinearAttention(dim_out, 
                                              attn_dim = layer_attn_dim_head, 
                                              heads = layer_attn_heads))
            
            if ind >= (len(in_out) - 1):
                layer.append(nn.Conv2d(dim_out, dim_in, 3, padding=1))
            else:
                layer.append(UpSamplingBlock(dim_out, dim_in))

            self.ups.append(layer)

        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = channels * (1 if not learned_variance else 2)

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim=time_emb_dim, dropout=dropout)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time_emb, x_self_cond=None):
        assert all([(d % self.downsample_factor) == 0 for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            if x_self_cond == None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            h.append(x)

            x = block2(x, time_emb)
            x = attn(x) + x
            h.append(x)
            
            x = downsample(x)
            
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, time_emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, time_emb)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, time_emb)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, time_emb)
        return self.final_conv(x)

if __name__ == "__main__":
    x = torch.zeros(10, 3, 32, 32)
    time = torch.zeros(10)
    
    device = 'cuda'
    from diffusion.nn.embedding import SinusoidalEmbedding
    time_emb = SinusoidalEmbedding(128)
    unet = Unet(3,
                init_dim=64, out_dim=3, time_emb_dim=128,
                dim_mults=(1, 2, 4, 8), channels=3, 
                self_condition=False, learned_variance=False, 
                dropout=0., attn_dim_head=32, 
                attn_heads=4, full_attn=None, flash_attn=False)
    
    time_emb = time_emb.to(device)
    unet = unet.to(device)
    x = x.to(device)
    time = time.to(device)
    
    te = time_emb(time)
    print(x.shape, te.shape)
    out = unet(x, te)
    print(out.shape)
        
    
    
