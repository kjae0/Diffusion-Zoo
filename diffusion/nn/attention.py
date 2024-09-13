from norms import RMSNorm

from einops import rearrange, repeat
from functools import partial

import torch
import torch.nn as nn

# TODO support flash attention
class Attention(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        if flash:
            raise NotImplementedError('flash attention not supported yet!')

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        if self.scale:
            scale = self.scale
        else:
            scale = q.shape[-1] ** -0.5

        # similarity
        sim = torch.einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = torch.einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class LinearAttention(nn.Module):
    def __init__(
        self,
        in_dim,
        heads = 4,
        attn_dim = 32,
        num_mem_kv = 4,
        norm_fn = RMSNorm
    ):
        super().__init__()
        self.scale = attn_dim ** -0.5
        self.heads = heads
        hidden_dim = attn_dim * heads

        self.norm = norm_fn(in_dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, attn_dim, num_mem_kv))
        self.to_qkv = nn.Conv2d(in_dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, in_dim, 1),
            RMSNorm(in_dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1) # B x (n_heads, attn_dim) x H x W
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)
    

class Attention2d(nn.Module):
    def __init__(
        self,
        in_dim,
        heads = 4,
        attn_dim = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = attn_dim * heads

        self.norm = RMSNorm(in_dim)
        self.attention = Attention(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, attn_dim))
        self.to_qkv = nn.Conv2d(in_dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, in_dim, 1)

    def forward(self, x):
        """
        Args:
            x : B x C x H x W

        Returns:
            out : B x C x H x W
        """
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attention(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)
