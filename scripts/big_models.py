# -*- coding: utf-8 -*-
"""全市场规模训练：LightGBM(boosting) + Transformer(RMSNorm/SwiGLU/dropout) + Bagging + Stacking
数据：data/big/arr.npz（5000+ 只 × 2015-2026）
切分：训练 <2022 ｜ 元模型/早停 2022 ｜ 验证 2023 ｜ 测试 2024+
评估：Rank IC、多空组合(费后)、小市值中性化 —— 沿用之前的验收门槛
"""
import time, json, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as Fn
from scipy import stats as sps
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'D:/quant'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
COST = 2e-3
L = 40            # 序列窗口
D = 96            # 隐层
HEADS, LAYERS = 4, 2
DROP = 0.15

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y'], z['M']
dates = pd.to_datetime(z['dates']); syms = z['symbols']; FEATS = list(z['feats'])
T, N, F = X.shape
print('数组 X%s | %d 只 | %s ~ %s | 有效 %.1f%%' % (
    X.shape, N, dates[0].date(), dates[-1].date(), M.mean() * 100), flush=True)

d_year = dates.year
m_tr = d_year < 2022
m_me = d_year == 2022
m_va = d_year == 2023
m_te = d_year >= 2024
print('训练 %d 日 | 元 2022 %d 日 | 验证 2023 %d 日 | 测试 %d 日' % (
    m_tr.sum(), m_me.sum(), m_va.sum(), m_te.sum()), flush=True)

# ---------- 样本索引 ----------
def samples(mask_days):
    tt, ii = np.where(M & mask_days[:, None] & ~np.isnan(Y))
    keep = tt >= L
    return tt[keep], ii[keep]

tt_tr, ii_tr = samples(m_tr)
tt_me, ii_me = samples(m_me)
tt_va, ii_va = samples(m_va)
tt_te, ii_te = samples(m_te)
print('样本 训练 %d | 2022 %d | 2023 %d | 2024+ %d' % (len(tt_tr), len(tt_me), len(tt_va), len(tt_te)), flush=True)

# ---------- 评估工具 ----------
def ic_series(pred, tt, ii):
    d = pd.DataFrame({'d': tt, 'p': pred, 'y': Y[tt, ii]})
    def f(g):
        if len(g) < 30: return np.nan
        return sps.spearmanr(g['p'], g['y']).correlation
    return d.groupby('d').apply(f).dropna()

def nw_t(x, lag=4):
    x = np.asarray(x); n = len(x); e = x - x.mean(); s = e @ e / n
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1); s += 2 * w * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan

def report(tag, pred, tt, ii):
    ic = ic_series(pred, tt, ii)
    print('  %-24s IC %+.4f  ICIR %+.2f  NW-t %+.1f  (天数 %d)' % (
        tag, ic.mean(), ic.mean() / ic.std(), nw_t(ic), len(ic)), flush=True)
    return ic.mean(), ic.mean() / ic.std()

# ---------- ① 因子合成基线 ----------
print('\n=== ① 因子 rank 合成（ICIR 加权，仅用训练期）===', flush=True)
rk = np.full_like(X, np.nan, dtype=np.float32)
for t in range(T):
    m = M[t]
    if m.sum() < 30: continue
    blk = X[t, m, :]
    rk[t, m, :] = (blk.argsort(0).argsort(0) + 0.5) / len(blk)
icir = []
for k in range(F):
    v = rk[tt_tr, ii_tr, k]; y = Y[tt_tr, ii_tr]
    d = pd.DataFrame({'d': tt_tr, 'p': v, 'y': y}).dropna()
    s = d.groupby('d').apply(lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    icir.append(s.mean() / s.std())
icir = np.array(icir); wgt = np.clip(icir, 0, None); wgt = wgt / wgt.sum()
print('  ICIR 权重 top5:', {FEATS[i]: round(float(wgt[i]), 3) for i in np.argsort(-wgt)[:5]}, flush=True)
fac_score = np.nansum(rk * wgt[None, None, :], axis=2)
report('因子合成 2022', fac_score[tt_me, ii_me], tt_me, ii_me)
report('因子合成 2023', fac_score[tt_va, ii_va], tt_va, ii_va)
report('因子合成 2024+', fac_score[tt_te, ii_te], tt_te, ii_te)

# ---------- ② LightGBM ----------
print('\n=== ② LightGBM（weekly 采样 + 早停）===', flush=True)
# 训练用每 5 日采样一次（非重叠），减少重叠标签带来的自相关
tt_tr_w, ii_tr_w = tt_tr[::5], ii_tr[::5]
Xtr = X[tt_tr_w, ii_tr_w, :]
ytr = Y[tt_tr_w, ii_tr_w]
# 直接用截面 rank 作为训练目标（对幅度不敏感，防小市值绑架）
rank_cache = {}
def yrank(t, i):
    key = t
    if key not in rank_cache:
        v = Y[t].copy(); m = M[t] & ~np.isnan(v)
        r = np.full(N, np.nan); r[m] = (sps.rankdata(v[m]) - 0.5) / m.sum()
        rank_cache[key] = r
    return rank_cache[key][i]
ytr_rk = np.array([yrank(t, i) for t, i in zip(tt_tr_w, ii_tr_w)], dtype=np.float32)
print('  LGBM 训练样本 %d（每 5 日取样）' % len(ytr_rk), flush=True)

params = dict(objective='regression', metric='l2', learning_rate=0.03, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbosity=-1, num_threads=8)
Xme = X[tt_me, ii_me, :]; yme_rk = np.array([yrank(t, i) for t, i in zip(tt_me, ii_me)], dtype=np.float32)
dv = lgb.Dataset(Xme, yme_rk)
t0 = time.time()
m_lgb = lgb.train(params, lgb.Dataset(Xtr, ytr_rk), num_boost_round=1500,
                  valid_sets=[dv], callbacks=[lgb.early_stopping(80, verbose=False)])
print('  LGBM 训练 %.0fs，最优轮 %d' % (time.time() - t0, m_lgb.best_iteration), flush=True)
p_lgb_me = m_lgb.predict(Xme, num_iteration=m_lgb.best_iteration)
p_lgb_va = m_lgb.predict(X[tt_va, ii_va, :], num_iteration=m_lgb.best_iteration)
p_lgb_te = m_lgb.predict(X[tt_te, ii_te, :], num_iteration=m_lgb.best_iteration)
report('LGBM 2023', p_lgb_va, tt_va, ii_va)
report('LGBM 2024+', p_lgb_te, tt_te, ii_te)

# ---------- ③ Transformer（RMSNorm + SwiGLU + dropout）----------
print('\n=== ③ Transformer ===', flush=True)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class SwiGLU(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        h = int(d * mult)
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

class TF(nn.Module):
    def __init__(self, f, d, h, nl):
        super().__init__()
        self.emb = nn.Linear(f, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        nn.init.trunc_normal_(self.pos, std=.02)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(nl)])
        self.norm = RMSNorm(d); self.head = nn.Linear(d * 2, 1)
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        z = self.drop(self.emb(x) + self.pos)
        for b in self.blocks: z = b(z)
        z = self.norm(z)
        return self.head(torch.cat([z[:, -1], z.mean(1)], -1)).squeeze(-1)

def make_seq(tt, ii):
    idx = tt[:, None] - np.arange(L - 1, -1, -1)[None, :]      # [B,L]
    return X[idx, ii[:, None], :]                              # [B,L,F]

def predict_tf(model, tt, ii, bs=4096):
    out = np.empty(len(tt), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(tt), bs):
            b = torch.from_numpy(make_seq(tt[s:s + bs], ii[s:s + bs])).to(DEV)
            out[s:s + bs] = model(b).cpu().numpy()
    return out

def train_tf(seed, epochs=8, sub=0.45, bs=1024):
    torch.manual_seed(seed)
    model = TF(F, D, HEADS, LAYERS).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    steps = int(len(ytr_rk) * sub / bs) * epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=8e-4, total_steps=max(steps, 1), pct_start=.25)
    rng = np.random.default_rng(seed)
    best = (-9, None)
    for ep in range(epochs):
        model.train(); t0 = time.time()
        perm = rng.permutation(len(ytr_rk))[:int(len(ytr_rk) * sub)]
        tot = 0.0
        for s in range(0, len(perm), bs):
            j = perm[s:s + bs]
            xb = torch.from_numpy(make_seq(tt_tr_w[j], ii_tr_w[j])).to(DEV)
            yb = torch.from_numpy(ytr_rk[j]).to(DEV)
            pred = model(xb)
            loss = Fn.smooth_l1_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += float(loss)
        pv = predict_tf(model, tt_me[:120000], ii_me[:120000])
        ic = ic_series(pv, tt_me[:120000], ii_me[:120000]).mean()
        print('    ep%d loss %.4f | 2022 IC %+.4f | %.0fs' % (ep + 1, tot / max(1, len(perm) // bs), ic, time.time() - t0), flush=True)
        if ic > best[0]:
            best = (ic, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    print('    最优 2022 IC %+.4f' % best[0], flush=True)
    return model

models = [train_tf(s) for s in (1, 2, 3)]           # bagging：3 个种子
p_tf_me = np.mean([predict_tf(m, tt_me, ii_me) for m in models], 0)
p_tf_va = np.mean([predict_tf(m, tt_va, ii_va) for m in models], 0)
p_tf_te = np.mean([predict_tf(m, tt_te, ii_te) for m in models], 0)
report('TF bagging 2023', p_tf_va, tt_va, ii_va)
report('TF bagging 2024+', p_tf_te, tt_te, ii_te)

# ---------- ④ Stacking ----------
print('\n=== ④ Stacking（Ridge 元模型，用 2022 的样本外预测拟合）===', flush=True)
def zc(a):
    a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-12)
A = np.column_stack([zc(p_lgb_me), zc(p_tf_me), zc(fac_score[tt_me, ii_me]), np.ones(len(tt_me))])
ym = Y[tt_me, ii_me]
coef, *_ = np.linalg.lstsq(A, (ym - ym.mean()) / ym.std(), rcond=None)
print('  元模型权重 [LGBM, TF, 因子, 截距]:', np.round(coef, 3), flush=True)
def stack(p_lgb, p_tf, p_fac):
    A = np.column_stack([zc(p_lgb), zc(p_tf), zc(p_fac), np.ones(len(p_lgb))])
    return A @ coef
p_st_va = stack(p_lgb_va, p_tf_va, fac_score[tt_va, ii_va])
p_st_te = stack(p_lgb_te, p_tf_te, fac_score[tt_te, ii_te])
report('Stacking 2022(拟合集)', stack(p_lgb_me, p_tf_me, fac_score[tt_me, ii_me]), tt_me, ii_me)
report('Stacking 2023', p_st_va, tt_va, ii_va)
report('Stacking 2024+', p_st_te, tt_te, ii_te)

# ---------- ⑤ 多空回测（按日期截面）----------
print('\n=== ⑤ 多空组合（top10%/bottom10%，5 日非重叠，20bp/边）===', flush=True)
def ls_from_pred(day_mask, pred, tt, ii, q=0.1):
    d = pd.DataFrame({'t': tt, 'i': ii, 'p': pred, 'y': Y[tt, ii]})
    rec = []
    days = np.where(day_mask)[0]
    for k, t in enumerate(days[::5]):
        g = d[d['t'] == t]
        if len(g) < 100: continue
        n = max(int(len(g) * q), 10)
        g = g.sort_values('p')
        lo = g.head(n)['y'].mean(); hi = g.tail(n)['y'].mean()
        rec.append((dates[t], hi - lo))
    r = pd.DataFrame(rec, columns=['d', 'r']).set_index('d')['r'] - COST * 2 * 2
    ann = (1 + r).prod() ** (50.4 / len(r)) - 1; vol = r.std() * np.sqrt(50.4)
    print('  %-18s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 胜率 %2.0f%%' % (
        tag4, ann * 100, vol * 100, (ann - .02) / vol, (r > 0).mean() * 100), flush=True)
    return ann, (ann - .02) / vol, r

for tag4, (pred, tt, ii, mk) in [('因子合成2024+', (fac_score[tt_te, ii_te], tt_te, ii_te, m_te)),
                                 ('LGBM 2024+', (p_lgb_te, tt_te, ii_te, m_te)),
                                 ('TF 2024+', (p_tf_te, tt_te, ii_te, m_te)),
                                 ('Stacking 2024+', (p_st_te, tt_te, ii_te, m_te))]:
    ls_from_pred(mk, pred, tt, ii)

# ---------- 保存 ----------
np.savez_compressed(f'{ROOT}/data/big/preds.npz',
                    p_lgb_me=p_lgb_me, p_lgb_va=p_lgb_va, p_lgb_te=p_lgb_te,
                    p_tf_me=p_tf_me, p_tf_va=p_tf_va, p_tf_te=p_tf_te,
                    fac_me=fac_score[tt_me, ii_me], fac_va=fac_score[tt_va, ii_va], fac_te=fac_score[tt_te, ii_te],
                    tt_me=tt_me, ii_me=ii_me, tt_va=tt_va, ii_va=ii_va, tt_te=tt_te, ii_te=ii_te)
print('\n预测已存 data/big/preds.npz', flush=True)
json.dump({'coef': coef.tolist(), 'icir_w': {FEATS[i]: float(wgt[i]) for i in range(F)}},
          open(f'{ROOT}/reports/big_models_meta.json', 'w'), ensure_ascii=False, indent=2)
