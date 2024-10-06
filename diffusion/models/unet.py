import sys
sys.path.append("/home/diya/Public/Image2Smiles/jy/Diffusion-Zoo")

from diffusion.nn.blocks import ResnetBlock, DownSamplingBlock, UpSamplingBlock
from diffusion.nn.attention import Attention2d, LinearAttention

from einops.layers.torch import Rearrange

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, image_dim, time_emb_dim, init_dim, 
                 out_dim=None, dim_mults=(1, 2, 4, 8),
                 self_condition=False, class_condition=False, num_classes=None,
                 learned_variance=False, dropout=0., attn_head_dim=32, attn_heads=4, 
                 full_attn=None, flash_attn=False
    ):
        super().__init__()
        self.self_condition = self_condition
        self.class_condition = class_condition
        
        input_dim = image_dim * (2 if self_condition else 1)
            
        if self.class_condition:
            if num_classes == None:
                raise ValueError('num_classes should be provided if class_condition is True')
            self.class_emb = nn.Embedding(num_classes, init_dim)
            self.null_class_emb = nn.Parameter(torch.randn(1, init_dim))
            
            classes_dim = time_emb_dim
            self.classes_mlp = nn.Sequential(
                nn.Linear(init_dim, classes_dim),
                nn.GELU(),
                nn.Linear(classes_dim, classes_dim)
            )
            
            # Will be concatenated to the input
            time_emb_dim += classes_dim

        self.init_conv = nn.Conv2d(input_dim, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 2)), True, False)

        num_stages = len(dim_mults)
        full_attn  = full_attn if isinstance(full_attn, tuple) else (full_attn,) * num_stages 
        attn_heads = attn_heads if isinstance(attn_heads, tuple) else (attn_heads,) * num_stages 
        attn_head_dim = attn_head_dim if isinstance(attn_head_dim, tuple) else (attn_head_dim,) * num_stages 
        
        assert len(full_attn) == len(dim_mults) == len(attn_heads) == len(attn_head_dim), 'length of full_attn, attn_heads, attn_head_dim should be equal to length of dim_mults'

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        ############################
        #    Downsampling blocks   #
        ############################
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_head_dim) in enumerate(zip(in_out, full_attn, attn_heads, attn_head_dim)):
            layer = nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_emb_dim, dropout = dropout),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_emb_dim, dropout = dropout)
            ])
            
            if layer_full_attn:
                layer.append(Attention2d(dim_in, 
                                       attn_dim = layer_attn_head_dim, 
                                       heads = layer_attn_heads,
                                       flash = flash_attn))
            else:
                layer.append(LinearAttention(dim_in, 
                                              attn_dim = layer_attn_head_dim, 
                                              heads = layer_attn_heads))
            
            if ind >= (num_resolutions - 1):
                layer.append(nn.Conv2d(dim_in, dim_out, 3, padding=1))
            else:
                layer.append(DownSamplingBlock(dim_in, dim_out))
                

            self.downs.append(layer)            

        ############################
        #       Middle blocks      #
        ############################
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim, dropout=dropout)
        self.mid_attn = Attention2d(mid_dim, heads=attn_heads[-1], attn_dim=attn_head_dim[-1], flash=flash_attn)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim, dropout=dropout)

        ############################
        #     Upsampling blocks    #
        ############################
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_head_dim) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_head_dim)))):
            layer = nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_emb_dim, dropout=dropout),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_emb_dim, dropout=dropout)
            ])
            
            if layer_full_attn:
                layer.append(Attention2d(dim_out, 
                                       attn_dim = layer_attn_head_dim, 
                                       heads = layer_attn_heads,
                                       flash = flash_attn))
            else:
                layer.append(LinearAttention(dim_out, 
                                              attn_dim = layer_attn_head_dim, 
                                              heads = layer_attn_heads))
            
            if ind >= (len(in_out) - 1):
                layer.append(nn.Conv2d(dim_out, dim_in, 3, padding=1))
            else:
                layer.append(UpSamplingBlock(dim_out, dim_in))

            self.ups.append(layer)

        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = image_dim * (1 if not learned_variance else 2)

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim=time_emb_dim, dropout=dropout)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)
        
        # self._init_weights()  # not working...
        
    def _init_weights(self):
        nn.init.xavier_normal_(self.init_conv.weight)
        nn.init.zeros_(self.init_conv.bias)
        
        for down in self.downs:
            for block in down:
                if isinstance(block, nn.Conv2d):
                    nn.init.xavier_normal_(block.weight)
                    nn.init.zeros_(block.bias)
                else:
                    block._init_weights()
                    
        for up in self.ups:
            for block in up:
                if isinstance(block, nn.Conv2d):
                    nn.init.xavier_normal_(block.weight)
                    nn.init.zeros_(block.bias)
                else:
                    block._init_weights()
                    
        nn.init.xavier_normal_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time_emb, class_cond=None, class_cond_mask=None, x_self_cond=None):
        assert all([(d % self.downsample_factor) == 0 for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        assert not(class_cond is None or not self.class_condition), 'Class can be provided if and only if class_condition is True'

        if self.self_condition:
            if x_self_cond == None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        if self.class_condition:
            if class_cond_mask is None:
                class_cond_mask = torch.ones_like(class_cond).bool()
                
            class_emb = self.class_emb(class_cond)
            class_emb = torch.where(class_cond_mask[:, None], class_emb, self.null_class_emb)        
            class_emb = self.classes_mlp(class_emb)
        else:
            class_emb = None

        x = self.init_conv(x)
        r = x.clone()

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb, class_emb)
            h.append(x)

            x = block2(x, time_emb, class_emb)
            x = attn(x) + x
            h.append(x)
            
            x = downsample(x)
            
        x = self.mid_block1(x, time_emb, class_emb)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, time_emb, class_emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, time_emb, class_emb)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, time_emb, class_emb)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, time_emb, class_emb)
        return self.final_conv(x)

if __name__ == "__main__":
    x = torch.zeros(10, 3, 32, 32)
    time = torch.zeros(10)
    classes = torch.randint(0, 10, (10,)).long()
    class_mask = torch.randn(10) > 0.5
    
    device = 'cuda:3'
    from diffusion.nn.embedding import SinusoidalEmbedding
    time_emb = SinusoidalEmbedding(128)
    unet = UNet(image_dim=3,
                init_dim=64, out_dim=3, time_emb_dim=128, class_condition=True, num_classes=10,
                dim_mults=(1, 2, 4, 8), 
                self_condition=False, learned_variance=False, 
                dropout=0., attn_head_dim=32, 
                attn_heads=4, full_attn=None, flash_attn=False)
    
    time_emb = time_emb.to(device)
    classes = classes.to(device)
    class_mask = class_mask.to(device)
    unet = unet.to(device)
    x = x.to(device)
    time = time.to(device)
    
    te = time_emb(time)
    print(x.shape, te.shape)
    out = unet(x, te, classes, class_cond_mask=class_mask)
    print(out.shape)
        