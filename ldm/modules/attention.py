from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import random
import time
import os
from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.diffusionmodules.util import normalization

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def otsu(mask_in):
    # normalize
    mask_norm = (mask_in - mask_in.min(-1, keepdim=True)[0]) / \
       (mask_in.max(-1, keepdim=True)[0] - mask_in.min(-1, keepdim=True)[0])
    
    bs = mask_in.shape[0]
    h = mask_in.shape[1]
    mask = []
    for i in range(bs):
        threshold_t = 0.
        max_g = 0.
        for t in range(10):
            mask_i = mask_norm[i]
            low = mask_i[mask_i < t*0.1]
            high = mask_i[mask_i >= t*0.1]
            low_num = low.shape[0]/h
            high_num = high.shape[0]/h
            low_mean = low.mean()
            high_mean = high.mean()
        
            g = low_num*high_num*((low_mean-high_mean)**2)
            if g > max_g:
                max_g = g
                threshold_t = t*0.1
            
        mask_i[mask_i < threshold_t] = 0
        mask_i[mask_i > threshold_t] = 1
        mask.append(mask_i)
    mask_out = torch.stack(mask, dim=0)
            
    return mask_out

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, return_attn=False):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        if return_attn:
            attn = rearrange(attn, '(b h) i j -> b h i j', h=h).mean(dim=1)
            return self.to_out(out), attn
        else:
            return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, return_attn=False, mask=None):
        return checkpoint(self._forward, (x, context, return_attn, mask), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, return_attn=False, mask=None):
        x = self.attn1(self.norm1(x)) + x
        if not return_attn:
            x = self.attn2(self.norm2(x), context=context, mask=mask) + x
            x = self.ff(self.norm3(x)) + x
            return x
        if return_attn:
            x_, attn = self.attn2(self.norm2(x), context=context, mask=mask, return_attn=True)
            x = x_ + x
            x = self.ff(self.norm3(x)) + x
            return x, attn
                  
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, image_cross=True):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.image_cross = image_cross
        
        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        if image_cross:
            self.image_cross_attention = BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout)
        
        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        
        self.timestamp = 0
        
    def forward(self, x, xr, context=None, cc_init=None, ph_pos=None, use_img_cond=True, return_attn=False):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        xr_in =xr
        
        x = self.norm(x)
        x = self.proj_in(x)
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # process reference images
        xr = self.norm(xr)
        xr = self.proj_in(xr)
        
        xr = rearrange(xr, 'b c h w -> b (h w) c')
        
        loss_reg = None
        ph_idx, eot_idx = ph_pos[0], ph_pos[1]
        
        for block in self.transformer_blocks:
            
            # weight = 0.7
            # context_w = context - F.one_hot(ph_pos+1, num_classes=77).unsqueeze(-1) * context * (1. - 1.)
            # context_w = context_w - F.one_hot(ph_pos, num_classes=77).unsqueeze(-1) * context_w * (1. - 0.5)
            
            attn_save = None
            x = block(x, context=context)
            xr, attn = block(xr, context=context, return_attn=True)
            
            if self.image_cross and use_img_cond:
                # get attention of xr
                # attn = block(xr, context=context, return_attn=True)
                attn = attn.transpose(1,2)
                attn_ph = attn[ph_idx].squeeze(1) # bs, n_patch
                attn_eot = attn[eot_idx].squeeze(1).detach()
                
                # ########################
                # attention reg
                if self.image_cross_attention.training:
                    loss_reg = F.mse_loss(attn_ph/attn_ph.max(-1, keepdim=True)[0], attn_eot/attn_eot.max(-1, keepdim=True)[0])
                # ########################
                    
                mask = attn_ph.detach()
                mask = otsu(mask).bool()
                
                x = self.image_cross_attention(x, context=xr, mask=mask)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        
        xr = rearrange(xr, 'b (h w) c -> b c h w', h=h, w=w)
        xr = self.proj_out(xr)
        
        return x + x_in, xr + xr_in, loss_reg, attn_save

        