# -*- coding: utf-8 -*-
"""Muon vs AdamW 严格 A/B（同 seed 同初始化、同数据顺序、同评估集）
Phase A：单 seed 快速扫 Muon 学习率（用 2022 IC 选，不碰 OOS）
Phase B：2 seed × 5 epoch 正式对照，报告 2023 / 2024+ OOS IC
"""
import time, json, math
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as Fn
from scipy import stats as sps
from muon_optimizer import HybridMuonAdam

ROOT = r'D:/quant'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
L, D, HEADS, LAYERS, DROP = 40, 96, 4, 2, 0.15
WD, BS = 1e-2, 1024

print('device', DEV, '| torch', torch.__version__, flush=True)
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates']); yr = dates.year; T, N, F = X.shape

def samp(mask):
    tt, ii = np.where(M & mask[:, None] & np.isfinite(Y))
    k = tt >= L
    return tt[k], ii[k]

def days_mask(years, step=5):
    d = np.where(np.isin(yr, years))[0]
    m = np.zeros(T, bool); m[d[::step]] = True
    return m

trd = np.where(yr < 2022)[0]
mtr = np.zeros(T, bool); mtr[trd[::5]] = True
tt_tr, ii_tr = samp(mtr)
ytr = pd.DataFrame({'t': tt_tr, 'y': Y[tt_tr, ii_tr]}).groupby('t')['y'].rank(pct=True).values.astype(np.float32)
tt_me, ii_me = samp(days_mask([2022]))
tt_va, ii_va = samp(days_mask([2023]))
tt_te, ii_te = samp(days_mask([2024, 2025, 2026]))
print('训练 %d 行 / %d 日 | 2022 %d | 2023 %d | 2024+ %d' %
      (len(tt_tr), len(np.unique(tt_tr)), len(tt_me), len(tt_va), len(tt_te)), flush=True)

# ---------------- 模型（与 big_train.py 完全一致） ----------------
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
        x = x + self.dp(a)                      # 必须用更新后的 x 再进 FFN
        return x + self.ff(self.n2(x))

class TFNet(nn.Module):
    def __init__(self, f, d, h, nl):
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

def make_seq(tt, ii):
    idx = tt[:, None] - np.arange(L - 1, -1, -1)[None, :]
    return X[idx, ii[:, None], :]

@torch.no_grad()
def pred(model, tt, ii, bs=4096):
    out = np.empty(len(tt), dtype=np.float32); model.eval()
    for s in range(0, len(tt), bs):
        b = torch.from_numpy(make_seq(tt[s:s + bs], ii[s:s + bs])).to(DEV)
        out[s:s + bs] = model(b).cpu().numpy()
    return out

def ic(p, tt, ii):
    d = pd.DataFrame({'d': tt, 'p': p, 'y': Y[tt, ii]})
    s = d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    return float(s.mean())

class OptWrap:
    """统一两种优化器接口"""
    def __init__(self, model, kind, lr, epochs):
        self.kind = kind; self.epochs = epochs
        if kind == 'adam':
            self.o = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
        else:
            self.o = HybridMuonAdam(model, lr_muon=lr, lr_adam=8e-4, weight_decay=WD)
    def zero_grad(self): self.o.zero_grad()
    def step(self): self.o.step()
    def sched(self, ep):
        if self.kind == 'adam':
            if not hasattr(self, '_sch'):
                self._sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.o, T_max=self.epochs, eta_min=8e-4 * .05)
            self._sch.step()
        else:
            self.o.cosine(ep + 1, self.epochs)

def run(tag, kind, lr, seed, epochs, sub, save_curve=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TFNet(F, D, HEADS, LAYERS).to(DEV)
    opt = OptWrap(model, kind, lr, epochs)
    rng = np.random.default_rng(seed)
    best = (-9, None, 0); curve = []
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(ytr))[:int(len(ytr) * sub)]
        tot = 0.0; nb = 0
        for s in range(0, len(perm), BS):
            j = perm[s:s + BS]
            xb = torch.from_numpy(make_seq(tt_tr[j], ii_tr[j])).to(DEV)
            yb = torch.from_numpy(ytr[j]).to(DEV)
            loss = Fn.smooth_l1_loss(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss); nb += 1
        opt.sched(ep)
        ic22 = ic(pred(model, tt_me, ii_me), tt_me, ii_me)
        curve.append(dict(ep=ep + 1, loss=tot / max(nb, 1), ic22=ic22))
        print('  %-16s seed%d ep%d loss %.5f | 2022 IC %+.4f | %.0fs' %
              (tag, seed, ep + 1, tot / max(nb, 1), ic22, time.time() - t0), flush=True)
        if ic22 > best[0]:
            best = (ic22, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    ic23 = ic(pred(model, tt_va, ii_va), tt_va, ii_va)
    ic24 = ic(pred(model, tt_te, ii_te), tt_te, ii_te)
    r = dict(phase=PH, tag=tag, kind=kind, lr=lr, seed=seed, epochs=epochs, sub=sub,
             best_ep=best[2], ic22=best[0], ic23=ic23, ic24=ic24,
             loss_final=curve[-1]['loss'], secs=round(time.time() - t0, 1))
    if save_curve:
        CURVES[tag + '_s%d' % seed] = curve
    print('  → %-16s best2022 %+.4f | 2023 %+.4f | 2024+ %+.4f | %.0fs\n' %
          (tag, best[0], ic23, ic24, r['secs']), flush=True)
    return r

if __name__ == '__main__':
    ROOT_ROWS = []; CURVES = {}

    # ---------------- Phase A：Muon 学习率扫描（只用 2022 选参） ----------------
    PH = 'A'
    print('\n=== Phase A｜Muon lr 扫描（seed=1, 3ep, sub=0.25，用 2022 IC 选参）===', flush=True)
    for tag, kind, lr in [('AdamW_8e4', 'adam', 8e-4), ('Muon_0.005', 'muon', .005),
                          ('Muon_0.02', 'muon', .02), ('Muon_0.05', 'muon', .05)]:
        ROOT_ROWS.append(run(tag, kind, lr, 1, 3, .25))

    pa = pd.DataFrame([r for r in ROOT_ROWS if r['phase'] == 'A' and r['kind'] == 'muon'])
    best_lr = float(pa.loc[pa['ic22'].idxmax(), 'lr'])
    print('Phase A 选出 Muon lr = %g（2022 IC %+.4f）\n' %
          (best_lr, pa['ic22'].max()), flush=True)

    # ---------------- Phase B：正式 A/B ----------------
    PH = 'B'
    print('=== Phase B｜正式 A/B（2 seed × 5ep × sub=0.5）===', flush=True)
    for seed in (1, 2):
        for tag, kind, lr in [('AdamW', 'adam', 8e-4), ('Muon', 'muon', best_lr)]:
            ROOT_ROWS.append(run(tag, kind, lr, seed, 5, .5))

    df = pd.DataFrame(ROOT_ROWS)
    df.to_csv(f'{ROOT}/reports/muon_ab.csv', index=False)
    json.dump(CURVES, open(f'{ROOT}/reports/muon_curves.json', 'w'), indent=1)

    print('\n=== 最终对比（Phase B，2024+ OOS）===', flush=True)
    b = df[df.phase == 'B']
    print(b.groupby('tag')[['ic22', 'ic23', 'ic24', 'loss_final', 'secs']].agg(['mean', 'std']).round(4).to_string())
    print('\nMuon 参数分组占比:', flush=True)
    _m = TFNet(F, D, HEADS, LAYERS)
    from muon_optimizer import split_params
    mp, ap = split_params(_m)
    print('  Muon %d 个张量 / %d 参数 (%.1f%%) | AdamW %d 个张量 / %d 参数' %
          (len(mp), sum(p.numel() for p in mp),
           sum(p.numel() for p in mp) / sum(p.numel() for p in _m.parameters()) * 100,
           len(ap), sum(p.numel() for p in ap)))
    print('\n已存 reports/muon_ab.csv、muon_curves.json')
