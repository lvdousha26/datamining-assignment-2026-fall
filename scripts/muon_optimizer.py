# -*- coding: utf-8 -*-
"""Muon 优化器（Newton–Schulz 正交化）+ AdamW 混合分组

参考 Keller Jordan 的 Muon / Moonlight 实现。要点：
  - 只对 2D 隐层权重矩阵做正交化更新；embedding 类、输出 head、bias、RMSNorm 增益、
    位置编码等仍交给 AdamW。
  - Muon 的更新量做了尺度归一化，因此 lr 要比 AdamW 大一个量级（经验 0.01~0.05）。
  - NS 迭代在 bfloat16 下做，5 步足够逼近 G (G^T G)^(-1/2)。

用法：
    opt = HybridMuonAdam(model, lr_muon=0.02, lr_adam=8e-4)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.cosine(ep, epochs)
"""
import math
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """五步 Newton–Schulz 把矩阵正交化（等价于 G @ (G^T G)^(-1/2) 的近似）"""
    X = G.to(torch.bfloat16)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95,
                 weight_decay=1e-2, ns_steps=5, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                     weight_decay=weight_decay,
                                     ns_steps=ns_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']; beta = group['momentum']
            wd = group['weight_decay']; ns = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                buf = st.get('buf')
                if buf is None:
                    buf = st['buf'] = torch.zeros_like(g)
                buf.lerp_(g, 1 - beta)
                upd = g + (buf - g) * beta if group['nesterov'] else buf
                u = zeropower_via_newtonschulz5(upd, ns)
                # 官方缩放：让不同形状矩阵的更新 RMS 大致一致
                u = u * max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.mul_(1 - lr * wd)
                p.add_(u, alpha=-lr)
        return loss


def split_params(model, exclude=('head',)):
    """2D 隐层矩阵 → Muon；其余（含输出 head、位置编码、norm、bias）→ AdamW"""
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not any(k in name for k in exclude):
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


class HybridMuonAdam:
    """把 Muon 与 AdamW 打包成和单个 optimizer 一样的接口，共享一个 cosine 调度"""

    def __init__(self, model, lr_muon=0.02, lr_adam=8e-4, momentum=0.95,
                 weight_decay=1e-2, ns_steps=5):
        mp, ap = split_params(model)
        self.muon = Muon(mp, lr=lr_muon, momentum=momentum,
                         weight_decay=weight_decay, ns_steps=ns_steps)
        self.adam = torch.optim.AdamW(ap, lr=lr_adam, weight_decay=weight_decay)
        self.lr_muon0, self.lr_adam0 = lr_muon, lr_adam
        self.n_muon = sum(p.numel() for p in mp)
        self.n_adam = sum(p.numel() for p in ap)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none)
        self.adam.zero_grad(set_to_none)

    def step(self):
        self.muon.step()
        self.adam.step()

    def cosine(self, ep, epochs, eta_min_ratio=0.05):
        f = eta_min_ratio + (1 - eta_min_ratio) * .5 * (1 + math.cos(math.pi * ep / max(epochs, 1)))
        for g in self.muon.param_groups:
            g['lr'] = self.lr_muon0 * f
        for g in self.adam.param_groups:
            g['lr'] = self.lr_adam0 * f
        return f
