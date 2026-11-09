# -*- coding: utf-8 -*-
"""TFNet 结构（与 big_train.py 完全一致），供实时打分 / 一次性验证共用"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

L, D, HEADS, LAYERS, DROP = 40, 96, 4, 2, 0.15


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__(); h = int(d * mult)
        self.w1 = nn.Linear(d, h); self.w3 = nn.Linear(d, h); self.w2 = nn.Linear(h, d)
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        return self.drop(self.w2(Fn.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1 = RMSNorm(d); self.att = nn.MultiheadAttention(d, h, dropout=DROP, batch_first=True)
        self.n2 = RMSNorm(d); self.ff = SwiGLU(d); self.dp = nn.Dropout(DROP)
    def forward(self, x):
        h = self.n1(x); a, _ = self.att(h, h, h, need_weights=False)
        x = x + self.dp(a)
        return x + self.ff(self.n2(x))


class TFNet(nn.Module):
    def __init__(self, f, d=D, h=HEADS, nl=LAYERS):
        super().__init__()
        self.emb = nn.Linear(f, d); self.pos = nn.Parameter(torch.zeros(1, L, d))
        nn.init.trunc_normal_(self.pos, std=.02)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(nl)])
        self.norm = RMSNorm(d); self.head = nn.Linear(d * 2, 1); self.drop = nn.Dropout(DROP)
    def forward(self, x):
        z = self.drop(self.emb(x) + self.pos)
        for b in self.blocks: z = b(z)
        z = self.norm(z)
        return self.head(torch.cat([z[:, -1], z.mean(1)], -1)).squeeze(-1)
