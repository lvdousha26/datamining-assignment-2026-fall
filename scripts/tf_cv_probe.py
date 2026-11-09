# -*- coding: utf-8 -*-
"""TF 的 CV 缺口探针：只跑最省钱的 2 折，回答"序列模型在小样本上会不会崩"

背景：5 折 Purged CV 里 LGBM / Ridge / 因子 / Stacking 都逐折重拟合了，只有 TF 没进 CV
（3 seed × 6 epoch × 5 折 ≈ 30 小时）。这在方法论上是真实缺口：如果 TF 在早期小样本
上失效，只有 CV 能发现。

本脚本用最小成本取样：折 1（训练 16.8 万行，2016-01~2017-07，全样本最少）与
折 5（训练 128 万行，对照）。各只训 1 个 seed、3 个 epoch。

口径声明
  - 训练/验证切分与 purged_cv.py 完全一致（purge 6 日，每 5 日采样）
  - 固定 epoch 数，**用最后一个 epoch 报告结果**；逐 epoch 的验证 IC 仅作诊断打印，
    不用于选模型（否则等于在验证集上早停，结果会系统性偏高）
  - 因此得到的数字是**下界**：正式训练用 3 seed bagging + 6 epoch，只会更好
"""
import time, json, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as Fn
from scipy import stats as sps
from tf_net import TFNet, L

ROOT = r'D:/quant'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
STEP, PURGE = 5, 6
BS, SUB, EPOCHS = 1024, 0.5, 3

ap = argparse.ArgumentParser()
ap.add_argument('--folds', default='1,5')
ap.add_argument('--epochs', type=int, default=EPOCHS)
ap.add_argument('--seeds', default='1')
ap.add_argument('--arch', default='tf', help='tf | noattn（同结构去掉注意力，只留 SwiGLU+残差）')
a = ap.parse_args()


class NoAttnNet(TFNet):
    """结构消融：保留 embedding / 位置编码 / RMSNorm / SwiGLU / 双池化，**只去掉注意力**。
    用来回答"IC 里有多少是 attention 贡献的，多少只是一个能看 40 天序列的 MLP"。"""

    def forward(self, x):
        z = self.drop(self.emb(x) + self.pos)
        for b in self.blocks:
            z = z + b.ff(b.n2(z))
        z = self.norm(z)
        return self.head(torch.cat([z[:, -1], z.mean(1)], -1)).squeeze(-1)

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates'])
T, N, F = X.shape
usable = np.where((M & np.isfinite(Y)).any(1))[0]
bounds = np.linspace(usable[0], usable[-1] + 1, 7).astype(int)
print('设备 %s | 折 %s | %d epoch | %d seed' % (DEV, a.folds, a.epochs, len(a.seeds.split(','))), flush=True)


def yrank(tt, ii):
    cache = {}; out = np.empty(len(tt), np.float32)
    for k, (t, i) in enumerate(zip(tt, ii)):
        if t not in cache:
            m = M[t] & np.isfinite(Y[t]); r = np.full(N, np.nan, np.float32)
            if m.sum() >= 30:
                v = Y[t][m]; r[m] = (sps.rankdata(v) - .5) / m.sum()
            cache[t] = r
        out[k] = cache[t][i]
    return out


def make_seq(tt, ii):
    idx = tt[:, None] - np.arange(L - 1, -1, -1)[None, :]
    return X[idx, ii[:, None], :]


@torch.no_grad()
def pred(model, tt, ii, bs=4096):
    model.eval(); out = np.empty(len(tt), np.float32)
    for s in range(0, len(tt), bs):
        b = torch.from_numpy(make_seq(tt[s:s + bs], ii[s:s + bs])).to(DEV)
        out[s:s + bs] = model(b).cpu().numpy()
    return out


def daily_ic(p, tt, ii):
    d = pd.DataFrame({'d': tt, 'p': p, 'y': Y[tt, ii]})
    return d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()


def train_one(tt, ii, ytr, seed, epochs, arch='tf'):
    torch.manual_seed(seed)
    net = (TFNet(F) if arch == 'tf' else NoAttnNet(F)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        net.train(); t0 = time.time()
        perm = rng.permutation(len(ytr))[:int(len(ytr) * SUB)]
        tot = 0.0
        for s in range(0, len(perm), BS):
            j = perm[s:s + BS]
            xb = torch.from_numpy(make_seq(tt[j], ii[j])).to(DEV)
            yb = torch.from_numpy(ytr[j]).to(DEV)
            loss = Fn.smooth_l1_loss(net(xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); tot += float(loss)
        sch.step()
        hist.append(tot / max(1, len(perm) // BS))
        print('    ep%d loss %.5f | %.0fs' % (ep + 1, hist[-1], time.time() - t0), flush=True)
    return net


rows = []
for k in [int(v) - 1 for v in a.folds.split(',')]:
    te0, te1 = bounds[k + 1], bounds[k + 2]
    tr0, tr1 = bounds[0], bounds[k + 1] - PURGE
    mtr = np.zeros(T, bool); mtr[tr0:tr1:STEP] = True
    mte = np.zeros(T, bool); mte[te0:te1:STEP] = True
    tt, ii = np.where(M & mtr[:, None] & np.isfinite(Y))
    ta, ia = np.where(M & mte[:, None] & np.isfinite(Y))
    keep = tt >= L
    tt, ii = tt[keep], ii[keep]
    print('\n折%d 训练 %s~%s (%d 行) → 验证 %s~%s (%d 行)' %
          (k + 1, dates[tr0].date(), dates[tr1 - 1].date(), len(tt),
           dates[te0].date(), dates[te1 - 1].date(), len(ta)), flush=True)
    ytr = yrank(tt, ii)
    for arch in a.arch.split(','):
        preds = []
        for s in a.seeds.split(','):
            net = train_one(tt, ii, ytr, int(s), a.epochs, arch)
            p = pred(net, ta, ia)
            s_ic = daily_ic(p, ta, ia)
            print('    [%s] seed%s 末轮验证 IC %+.4f (ICIR %.2f)' %
                  (arch, s, s_ic.mean(), s_ic.mean() / s_ic.std()), flush=True)
            preds.append(p)
        pb = np.mean(preds, 0)
        sc = daily_ic(pb, ta, ia)
        rows.append(dict(fold=k + 1, arch=arch, n_train=len(tt), n_val=len(ta),
                         val_days=len(sc), ic=sc.mean(), icir=sc.mean() / sc.std(),
                         epochs=a.epochs, seeds=len(preds),
                         train_to=str(dates[tr1 - 1].date()), val_to=str(dates[te1 - 1].date())))
        print('  → 折%d %s(%d seed, %d epoch) IC %+.4f ICIR %.2f | 同折 LGBM 见 purged_cv_folds.csv' %
              (k + 1, arch, len(preds), a.epochs, sc.mean(), sc.mean() / sc.std()), flush=True)
        del preds

d = pd.DataFrame(rows)
d.to_csv(f'{ROOT}/reports/tf_cv_probe.csv', index=False)
print('\n=== 汇总 ===')
print(d[['fold', 'arch', 'n_train', 'val_days', 'ic', 'icir']].round(4).to_string(index=False))
if 'tf' in d.arch.values and 'noattn' in d.arch.values:
    print('\n结构消融（去注意力 − 保留注意力）：')
    for f, g in d.groupby('fold'):
        t = g[g.arch == 'tf'].ic.iloc[0]; n = g[g.arch == 'noattn'].ic.iloc[0]
        print('  折%d  TF %+.4f | NoAttn %+.4f | attention 贡献 %+.4f (%.1f%%)' %
              (f, t, n, t - n, (t - n) / abs(n) * 100))
print('已存 reports/tf_cv_probe.csv')
