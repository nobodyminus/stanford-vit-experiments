import torch
from einops import rearrange
from torch import nn
from enum import Enum


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MaxAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., routes=4):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # Create a modulelist for each of our routes
        self.qkvs = nn.ModuleList([nn.Linear(dim, inner_dim * 3, bias=False) for _ in range(routes)])
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = [layer(x) for layer in self.qkvs]
        qkv = torch.stack(qkv, dim=-2)
        qkv = qkv.chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n a (h d) -> b h a n d', h=self.heads), qkv)

        # Throw out all but the first expansion of v, work out a more efficient way to do this
        v = v[:, :, 0, :, :]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        # Take maximum, as suggested
        attn = torch.max(attn, dim=-3)[0]
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class VectorMessage(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., routes=4):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # Create a modulelist to hold our routes
        self.qkvs = nn.ModuleList([nn.Linear(dim, inner_dim * 3, bias=False) for _ in range(routes)])
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # Pass through routes and stack
        qkv = [layer(x) for layer in self.qkvs]
        qkv = torch.stack(qkv, dim=-2)
        qkv = qkv.chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n a (h d) -> b h a n d', h=self.heads), qkv)

        # Throw out all but the first expansion of v, work out a more efficient way to do this
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        # Aggregate messages to preserve size
        out = torch.squeeze(torch.mean(out, dim=-3))
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class AttentionSelector(Enum):
    BASELINE = Attention
    MAX = MaxAttention
    MESSAGE = VectorMessage
