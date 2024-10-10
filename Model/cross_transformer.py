import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pdb
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, dim,hidden_dim,num_query_token, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.num_query_token=num_query_token
        self.query_token = nn.Parameter(torch.randn(1, num_query_token, dim))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, hidden_dim, dropout = dropout)
            ]))

    def forward(self,x_1,x_2):
        b=x_1.size(0)
        N=x_1.size(2)
        
        x_1=x_1.permute(0,2,1)
        x_2=x_2.permute(0,2,1)
        for attn_1, ff in self.layers:
            x_12 = torch.cat((x_1, x_2), dim = 1)
            out_1 = attn_1(x_1, x_12, kv_include_self = False) + x_1 
            out_1 = ff(out_1) + out_1
            x_1 = out_1   
        x_1=x_1.permute(0,2,1)
          
        return x_1

