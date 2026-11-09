# -*- coding: utf-8 -*-
"""全市场规模训练：LightGBM(boosting) + Transformer(RMSNorm/SwiGLU/dropout) + Bagging + Stacking(Ridge)
数据：data/big/arr.npz（5378 只 × 2616 日 × 42 因子）
切分：train <2024 ｜ meta（早停 / Stacking 拟合）2024 ｜ dev（验证 / 调参）2025H1
      ｜ lockbox 2025-07 起（标签物理隔离，只用一次）
评估：Rank IC、多空组合(费后)
"""
import time, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as Fn
from scipy import stats as sps
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ---- 运行日志：统一写 logs/，不改变原有输出 ----
if __name__ == '__main__':
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from runlog import RunLogger as _RL
        _RL(__file__).__enter__()
    except Exception:
        pass

ROOT = r'D:/quant'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
COST, L, D, HEADS, LAYERS, DROP = 2e-3, 40, 96, 4, 2, 0.15

from muon_optimizer import HybridMuonAdam
# Muon 开关。A/B 结论（reports/muon_ab.md）：充分训练后 Muon 与 AdamW 的
# 2024+ OOS IC 持平（0.1012 vs 0.1009），但 Muon seed 间波动大 8 倍、慢约 10%，
# 故默认不开。适用场景：训练预算受限 / 只跑 1~2 epoch —— Muon 第 2 个 epoch
# 就达到 AdamW 第 5 个 epoch 的验证 IC，可省约 60% 算力。
# 另：AdamW lr 不要从 8e-4 往上调（3e-3 在充分训练下 OOS 0.0988 < 0.1009），
# 短训练下看似涨点只是收敛更快，见 reports/muon_report.md。
USE_MUON, MUON_LR = False, 5e-3

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, M = z['X'], z['M']
Y = z['Y2']          # 可执行口径：t 日收盘出信号 → t+1 收盘买入 → 持有 5 日
dates = pd.to_datetime(z['dates']); syms = z['symbols']; FEATS = list(z['feats'])
T, N, F = X.shape
print('数组 %s | %d 只 | %s ~ %s | 覆盖 %.1f%%' % (
    X.shape, N, dates[0].date(), dates[-1].date(), M.mean() * 100), flush=True)

LOCK_START = str(z['lock_start']) if 'lock_start' in z else '2025-07-01'
EMBARGO = 6          # 标签用到 t+6，故隔离 6 个交易日（5 日重叠标签 + 1 日执行延迟）
day_mask = {'tr': dates < '2024-01-01',
            'me': (dates >= '2024-01-01') & (dates < '2025-01-01'),
            # 注意：va 与 te 目前是同一段（2025H1）。历史脚本把它当成"dev-val / dev-OOS"
            # 两段用，实际指向同一批样本，报告里出现的两个数字不是两次独立验证。
            # 真正的样本外只有 lockbox（2025-07 起）。若要拆成前半/后半，需重跑全流程。
            'va': (dates >= '2025-01-01') & (dates < LOCK_START),
            'te': (dates >= '2025-01-01') & (dates < LOCK_START)}

def embargo(mask, e=EMBARGO):
    """剔除每段最后 e 个交易日，防止 5 日重叠标签跨切分边界泄漏"""
    out = mask.copy(); idx = np.where(out)[0]
    if len(idx) > e: out[idx[-e:]] = False
    return out

day_mask = {k: embargo(v) for k, v in day_mask.items()}
print('切分（含 %d 日 embargo，lockbox %s 起完全不参与）:' % (EMBARGO, LOCK_START),
      {k: int(v.sum()) for k, v in day_mask.items()}, '（交易日数）', flush=True)

def samples(md):
    tt, ii = np.where(M & md[:, None] & np.isfinite(Y))
    k = tt >= L
    return tt[k], ii[k]

tr_days = np.where(day_mask['tr'])[0]
sel_days = tr_days[np.arange(0, len(tr_days), 5)]
md_tr_w = np.zeros(T, bool); md_tr_w[sel_days] = True
tt_tr, ii_tr = samples(md_tr_w)
tt_me, ii_me = samples(day_mask['me'])
tt_va, ii_va = samples(day_mask['va'])
tt_te, ii_te = samples(day_mask['te'])
print('样本 train %d（%d 日）| meta %d | dev-val %d | dev-OOS %d' % (
    len(tt_tr), len(sel_days), len(tt_me), len(tt_va), len(tt_te)), flush=True)

def rank_mat(blk, m):
    """blk [N,F] 逐列截面百分位；返回同形状（无效行为 NaN）"""
    r = np.full_like(blk, np.nan, dtype=np.float32)
    if m.sum() < 30: return r
    sub = blk[m]                                  # [n,F]
    r[m] = ((sub.argsort(0).argsort(0) + 0.5) / len(sub)).astype(np.float32)
    return r

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
    print('  %-26s IC %+.4f  ICIR %+.2f  NW-t %+.1f' % (
        tag, ic.mean(), ic.mean() / ic.std(), nw_t(ic)), flush=True)
    return ic.mean()

def yrank_pairs(tt, ii):
    cache = {}
    out = np.empty(len(tt), dtype=np.float32)
    for k, (t, i) in enumerate(zip(tt, ii)):
        if t not in cache:
            m = M[t] & np.isfinite(Y[t])
            r = np.full(N, np.nan, dtype=np.float32)
            if m.sum() >= 30:
                v = Y[t][m]; r[m] = (sps.rankdata(v) - 0.5) / m.sum()
            cache[t] = r
        out[k] = cache[t][i]
    return out

# ---------- ① 因子合成 ----------
print('\n=== ① 因子 rank 合成（ICIR 加权，只用 train 段）===', flush=True)
ic_sum = np.zeros(F); ic_cnt = np.zeros(F)
for t in tr_days:
    m = M[t] & np.isfinite(Y[t])
    if m.sum() < 30: continue
    rk = rank_mat(X[t], m)                        # [N,F]
    yv = Y[t][m]
    for k in range(F):
        v = rk[m, k]
        if np.std(v) > 0:
            c = sps.spearmanr(v, yv).correlation
            if np.isfinite(c):
                ic_sum[k] += c; ic_cnt[k] += 1
ic_per_f = ic_sum / np.maximum(ic_cnt, 1)
wgt = np.clip(ic_per_f, 0, None); wgt = wgt / wgt.sum()
print('  因子 IC:', {FEATS[i]: round(float(ic_per_f[i]), 4) for i in np.argsort(-ic_per_f)[:6]}, flush=True)
print('  权重 top5:', {FEATS[i]: round(float(wgt[i]), 3) for i in np.argsort(-wgt)[:5]}, flush=True)

fac = np.full((T, N), np.nan, dtype=np.float32)
for t in range(T):
    m = M[t]
    if m.sum() < 30: continue
    rk = rank_mat(X[t], m)
    fac[t] = np.nan_to_num(rk) @ wgt
fac[~M] = np.nan

report('因子合成 meta', fac[tt_me, ii_me], tt_me, ii_me)
report('因子合成 dev-val', fac[tt_va, ii_va], tt_va, ii_va)
report('因子合成 dev-OOS', fac[tt_te, ii_te], tt_te, ii_te)

# ---------- ② LightGBM ----------
print('\n=== ② LightGBM（每 5 日采样，meta 段早停）===', flush=True)
ytr = yrank_pairs(tt_tr, ii_tr)
yme = yrank_pairs(tt_me, ii_me)
params = dict(objective='regression', metric='l2', learning_rate=0.03, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbosity=-1, num_threads=8)
t0 = time.time()
m_lgb = lgb.train(params, lgb.Dataset(X[tt_tr, ii_tr, :], ytr), num_boost_round=1200,
                  valid_sets=[lgb.Dataset(X[tt_me, ii_me, :], yme)],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
print('  LGBM %.0fs 最优轮 %d' % (time.time() - t0, m_lgb.best_iteration), flush=True)
ni = m_lgb.best_iteration
import os, json as _json
CK = f'{ROOT}/checkpoints'; os.makedirs(CK, exist_ok=True)
m_lgb.save_model(f'{CK}/lgbm_big.txt')
_json.dump({'best_iter': int(ni), 'params': params, 'features': FEATS,
            'label': 'Y2 可执行口径'}, open(f'{CK}/lgbm_meta.json', 'w'), ensure_ascii=False, indent=2)
print('  LGBM 已存盘 checkpoints/lgbm_big.txt', flush=True)
p_lgb = {k: m_lgb.predict(X[tt, ii, :], num_iteration=ni)
         for k, (tt, ii) in [('me', (tt_me, ii_me)), ('va', (tt_va, ii_va)), ('te', (tt_te, ii_te))]}
report('LGBM dev-val', p_lgb['va'], tt_va, ii_va)
report('LGBM dev-OOS', p_lgb['te'], tt_te, ii_te)

# ---------- ③ Transformer ----------
print('\n=== ③ Transformer (RMSNorm+SwiGLU+dropout, 3 seed) ===', flush=True)

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
        h = self.n1(x)
        a, _ = self.att(h, h, h, need_weights=False)
        x = x + self.dp(a)
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

def pred_tf(model, tt, ii, bs=4096):
    out = np.empty(len(tt), dtype=np.float32); model.eval()
    with torch.no_grad():
        for s in range(0, len(tt), bs):
            b = torch.from_numpy(make_seq(tt[s:s + bs], ii[s:s + bs])).to(DEV)
            out[s:s + bs] = model(b).cpu().numpy()
    return out

def train_tf(seed, epochs=6, sub=0.5, bs=1024):
    torch.manual_seed(seed)
    model = TFNet(F, D, HEADS, LAYERS).to(DEV)
    if USE_MUON:
        # 2D 隐层矩阵走 Muon（NS 正交化），head / norm / bias / 位置编码仍走 AdamW
        opt = HybridMuonAdam(model, lr_muon=MUON_LR, lr_adam=8e-4,
                             momentum=0.95, weight_decay=1e-2)
        sch = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    rng = np.random.default_rng(seed)
    best = (-9, None)
    for ep in range(epochs):
        model.train(); t0 = time.time()
        perm = rng.permutation(len(ytr))[:int(len(ytr) * sub)]
        tot = 0.0
        for s in range(0, len(perm), bs):
            j = perm[s:s + bs]
            xb = torch.from_numpy(make_seq(tt_tr[j], ii_tr[j])).to(DEV)
            yb = torch.from_numpy(ytr[j]).to(DEV)
            loss = Fn.smooth_l1_loss(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss)
        if USE_MUON:
            opt.cosine(ep + 1, epochs, eta_min_ratio=0.0125)   # 与 AdamW 的 eta_min=1e-5 对齐
        else:
            sch.step()
        ic = ic_series(pred_tf(model, tt_me, ii_me), tt_me, ii_me).mean()
        print('    seed%d ep%d loss %.5f | 2022 IC %+.4f | %.0fs' % (
            seed, ep + 1, tot / max(1, len(perm) // bs), ic, time.time() - t0), flush=True)
        if ic > best[0]: best = (ic, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    print('    seed%d 最优 2022 IC %+.4f' % (seed, best[0]), flush=True)
    return model

tfs = []
for _s in (1, 2, 3):
    _m = train_tf(_s)
    torch.save(_m.state_dict(), f'{CK}/tf_seed{_s}.pt')      # 断点：每个种子单独存
    print('    seed%d 权重已存 checkpoints/tf_seed%d.pt' % (_s, _s), flush=True)
    tfs.append(_m)
p_tf = {k: np.mean([pred_tf(m, tt, ii) for m in tfs], 0)
        for k, (tt, ii) in [('me', (tt_me, ii_me)), ('va', (tt_va, ii_va)), ('te', (tt_te, ii_te))]}
report('TF bagging meta', p_tf['me'], tt_me, ii_me)
report('TF bagging dev-val', p_tf['va'], tt_va, ii_va)
report('TF bagging dev-OOS', p_tf['te'], tt_te, ii_te)

# ---------- ④ Stacking ----------
print('\n=== ④ Stacking（真正的 Ridge：L2 + 截距不惩罚，标准化参数固定在 meta 段）===', flush=True)
from stack_ridge import stats_of, design, fit_ridge, select_lambda, fix_scale

p_me = (p_lgb['me'], p_tf['me'], fac[tt_me, ii_me])
MU, SD = stats_of(*p_me)                       # 只用 meta 段估计，之后固定
A = design(p_me, MU, SD)
ym = Y[tt_me, ii_me].astype(np.float64)
yz = (ym - ym.mean()) / ym.std()
LAM, _tab = select_lambda(A, yz, tt_me)        # 段内时序选正则强度，不看验证段
print(_tab.to_string(index=False), flush=True)
coef = fix_scale(fit_ridge(A, yz, LAM), A)     # 缩放到 meta 段预测 std=1（不影响排序）
print('  正则 lam=%g | 权重 [LGBM, TF, 因子, 截距]: %s' % (LAM, np.round(coef, 4)), flush=True)
p_st = {k: design((p_lgb[k], p_tf[k], fac[tt, ii]), MU, SD) @ coef
        for k, (tt, ii) in [('me', (tt_me, ii_me)), ('va', (tt_va, ii_va)), ('te', (tt_te, ii_te))]}
report('Stacking meta(拟合)', p_st['me'], tt_me, ii_me)
report('Stacking dev-val', p_st['va'], tt_va, ii_va)
report('Stacking dev-OOS', p_st['te'], tt_te, ii_te)

# ---------- ⑤ 多空 ----------
print('\n=== ⑤ 多空（top/bottom 10%，5 日非重叠，双向 20bp）===', flush=True)
def ls(pred, tt, ii, md, q=0.1, tag=''):
    d = pd.DataFrame({'t': tt, 'i': ii, 'p': pred, 'y': Y[tt, ii]})
    rec = []
    for t in np.where(md)[0][::5]:
        g = d[d['t'] == t]
        if len(g) < 100: continue
        n = max(int(len(g) * q), 10)
        gs = g.sort_values('p')
        rec.append((dates[t], gs.tail(n)['y'].mean() - gs.head(n)['y'].mean()))
    r = pd.DataFrame(rec, columns=['d', 'r']).set_index('d')['r'] - COST * 4
    ann = (1 + r).prod() ** (50.4 / len(r)) - 1; vol = r.std() * np.sqrt(50.4)
    print('  %-14s 年化 %7.2f%% 波动 %5.2f%% 夏普 %5.2f 胜率 %2.0f%%' % (
        tag, ann * 100, vol * 100, (ann - .02) / vol, (r > 0).mean() * 100), flush=True)
    return ann, (ann - .02) / vol, r

for tag, pred in [('因子合成', fac[tt_te, ii_te]), ('LGBM', p_lgb['te']),
                  ('TF', p_tf['te']), ('Stacking', p_st['te'])]:
    ls(pred, tt_te, ii_te, day_mask['te'], tag=tag)

np.savez(f'{ROOT}/data/big/preds.npz',
         p_lgb_me=p_lgb['me'], p_lgb_va=p_lgb['va'], p_lgb_te=p_lgb['te'],
         p_tf_me=p_tf['me'], p_tf_va=p_tf['va'], p_tf_te=p_tf['te'],
         p_st_me=p_st['me'], p_st_va=p_st['va'], p_st_te=p_st['te'],
         fac_me=fac[tt_me, ii_me], fac_va=fac[tt_va, ii_va], fac_te=fac[tt_te, ii_te],
         tt_me=tt_me, ii_me=ii_me, tt_va=tt_va, ii_va=ii_va, tt_te=tt_te, ii_te=ii_te)
json.dump({'coef': coef.tolist(), 'coef_mu': MU.tolist(), 'coef_sd': SD.tolist(),
           'ridge_lambda': LAM, 'stack_model': 'ridge(L2, 截距不惩罚, 固定 meta 标准化)',
           'lock_start': LOCK_START, 'embargo': EMBARGO,
           'coef_order': ['p_lgb', 'p_tf', 'fac', 'intercept'],
           'feature_order': FEATS, 'factor_ic': {FEATS[i]: float(ic_per_f[i]) for i in range(F)},
           'weights': {FEATS[i]: float(wgt[i]) for i in range(F)}},
          open(f'{ROOT}/reports/big_models_meta.json', 'w'), ensure_ascii=False, indent=2)
print('\n预测已存 data/big/preds.npz', flush=True)
