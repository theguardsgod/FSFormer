import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Filter(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        cls  = x.permute(1,0,2)[0].unsqueeze(1)
        
        x = self.fn(x, **kwargs) * x.permute(1,0,2)[1:].permute(1,0,2)
        x = torch.cat([cls, x], dim=1)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,

        dropout = 0.
    ):
        super().__init__()
        dim_head = dim / heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return out

class FSAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        dim_head = dim / heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_q = nn.Linear(dim, dim, bias = False)
        
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, 1, bias = False)

    def forward(self, x):
        h = self.heads
        cls = x.permute(1,0,2)[0].unsqueeze(1)
        x = x.permute(1,0,2)[1:].permute(1,0,2)
        k, v = self.to_kv(cls).chunk(2, dim = -1)
        q = self.to_q(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # print(q.shape,k.shape,v.shape)
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        
        #attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.out(out)
        return self.relu(out)

# transformer

class FSFormer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, attn_dropout, ff_dropout):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout = ff_dropout))),
                Filter(PreNorm(dim, FSAttention(dim, heads = heads, dropout = attn_dropout))),
            ]))

    def forward(self, x):
        
        
        for attn, ff, fsattn in self.layers:
            x = attn(x)
            x = ff(x)
            x = fsattn(x)
        return x
# mlp

class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# main class

class TabTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_features,
        dim,
        depth,
        heads,
        mlp_act = nn.ReLU(),
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        # categories related calculations
        self.num_features = num_features
        self.embeds = nn.Linear(1, dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        # transformer
        self.transformer = FSFormer(
            num_tokens = self.num_features,
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )
        self.fc1 = nn.Linear(num_features+1, 50)
        self.fc2 = nn.Linear(50, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        
        self.pooling = nn.AdaptiveAvgPool2d((None, 1))
        self.bn = nn.BatchNorm1d(50)
        self.sig  = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = x.unsqueeze(2)
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)[0].unsqueeze(0).permute(1, 0, 2)
        
        # out = F.relu(x, inplace=True)
        
        # out = torch.flatten(out, 1)
        # Embedding the features using FC
        x = x.unsqueeze(2)
        x = self.embeds(x)
        # Get the CLS token by average pooling
        
        cls_token = self.avgpool(x)
        x = torch.cat([cls_token, x], dim=1)

        x = self.transformer(x)
        
        

        x = self.pooling(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        # 
        # out = F.relu(x, inplace=True)
        
        # out = torch.flatten(out, 1)
        

        # out = self.classifier1(out)
        # self.sig(out)
        # out = self.classifier2(out)
        
        return self.sig(x)