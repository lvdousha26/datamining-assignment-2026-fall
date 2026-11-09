# -*- coding: utf-8 -*-
"""过拟合审计：IS/OOS 衰减 · 标签安慰剂 · 超参敏感度 · 分年度稳定性
统一子样本，所有模型可比；评估集固定为 2024+ 每 5 日一个截面。
"""
import numpy as np, pandas as pd, lightgbm as lgb, time, json
from scipy import stats as sps

ROOT = r'D:/quant'
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates']); yr = dates.year; T, N, F = X.shape
pr = np.load(f'{ROOT}/data/big/preds.npz')
NI = json.load(open(f'{ROOT}/checkpoints/lgbm_meta.json'))['best_iter']
BASE = dict(objective='regression', metric='l2', learning_rate=0.03, num_leaves=63,
            min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=8, seed=42)

def samp(mask):
    tt, ii = np.where(M & mask[:, None] & np.isfinite(Y))
    return tt, ii

def ic_pairs(p, tt, ii, y=None):
    v = Y[tt, ii] if y is None else y
    d = pd.DataFrame({'d': tt, 'p': p, 'y': v})
    s = d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    return s.mean(), s.std()

def yrank(tt, ii):
    cache = {}; out = np.empty(len(tt), dtype=np.float32)
    for k, (t, i) in enumerate(zip(tt, ii)):
        if t not in cache:
            m = M[t] & np.isfinite(Y[t]); r = np.full(N, np.nan, dtype=np.float32)
            if m.sum() >= 30:
                v = Y[t][m]; r[m] = (sps.rankdata(v) - .5) / m.sum()
            cache[t] = r
        out[k] = cache[t][i]
    return out

# 训练样本：与正式训练同口径（<2022，每 5 日），取 30 万行子样本
trd = np.where(yr < 2022)[0]; mtr = np.zeros(T, bool); mtr[trd[::5]] = True
tt_tr, ii_tr = samp(mtr)
rng = np.random.default_rng(0)
k = rng.choice(len(tt_tr), min(300000, len(tt_tr)), replace=False)
ts, is_ = tt_tr[k], ii_tr[k]
ys = yrank(ts, is_)
# 评估集：2024+ 每 5 日
ted = np.where(yr >= 2024)[0]; mte = np.zeros(T, bool); mte[ted[::5]] = True
ta, ia = samp(mte)
Xa = X[ta, ia, :]
print('训练子样本 %d 行 | OOS 评估 %d 行 / %d 日 | best_iter=%d' %
      (len(ts), len(ta), len(np.unique(ta)), NI), flush=True)

# ---------- ① 样本内 vs 样本外 ----------
print('\n=== ① IS / OOS 衰减（已保存的正式 LGBM）===', flush=True)
m0 = lgb.Booster(model_file=f'{ROOT}/checkpoints/lgbm_big.txt')
p_is = m0.predict(X[ts, is_, :], num_iteration=NI)
a, _ = ic_pairs(p_is, ts, is_)
p_oos = m0.predict(Xa, num_iteration=NI)
b, _ = ic_pairs(p_oos, ta, ia)
print('  训练集内 IC %+.4f | 2024+ OOS IC %+.4f | 衰减 %.0f%%' % (a, b, (1 - b / a) * 100))

# ---------- ② 分年度模型 IC ----------
print('\n=== ② 分年度 Stacking IC（用已落盘预测）===', flush=True)
tt = np.concatenate([pr['tt_me'], pr['tt_va'], pr['tt_te']])
ii = np.concatenate([pr['ii_me'], pr['ii_va'], pr['ii_te']])
pp = np.concatenate([pr['p_st_me'], pr['p_st_va'], pr['p_st_te']])
yy = Y[tt, ii]
for y in [2022, 2023, 2024, 2025, 2026]:
    sel = yr[tt] == y
    if sel.sum() < 1000: continue
    d = pd.DataFrame({'d': tt[sel], 'p': pp[sel], 'y': yy[sel]})
    s = d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    print('  %d  IC %+.4f  ICIR %+.2f  日数 %d' % (y, s.mean(), s.mean() / s.std(), len(s)))

# ---------- ③ 标签安慰剂 ----------
print('\n=== ③ 安慰剂：训练标签逐日打乱（同超参、同样本）===', flush=True)
perm = ys.copy()
dfp = pd.DataFrame({'d': ts, 'y': ys})
for d, g in dfp.groupby('d', sort=False):
    perm[g.index.values] = rng.permutation(g['y'].values)
t0 = time.time()
mp = lgb.train(BASE, lgb.Dataset(X[ts, is_, :], perm), num_boost_round=NI)
vp, _ = ic_pairs(mp.predict(Xa, num_iteration=NI), ta, ia)
mr = lgb.train(BASE, lgb.Dataset(X[ts, is_, :], ys), num_boost_round=NI)
vr, _ = ic_pairs(mr.predict(Xa, num_iteration=NI), ta, ia)
print('  真标签 OOS IC %+.4f | 打乱标签 OOS IC %+.4f | 差 %+.4f  (%.0fs)' %
      (vr, vp, vr - vp, time.time() - t0))

# ---------- ④ 超参敏感度 ----------
print('\n=== ④ 超参敏感度（8 组，同子样本 / %d 轮）===' % NI, flush=True)
grid = [(nl, md) for nl in (15, 31, 63, 127) for md in (100, 500)]
rows = []
t0 = time.time()
for nl, md in grid:
    P = dict(BASE); P['num_leaves'] = nl; P['min_data_in_leaf'] = md
    m = lgb.train(P, lgb.Dataset(X[ts, is_, :], ys), num_boost_round=NI)
    v, _ = ic_pairs(m.predict(Xa, num_iteration=NI), ta, ia)
    rows.append((nl, md, v))
    print('  leaves=%3d min_data=%3d → OOS IC %+.4f' % (nl, md, v), flush=True)
vals = [r[2] for r in rows]
print('  OOS IC 范围 %+.4f ~ %+.4f | 标准差 %.4f | 极差 %.4f (%.0fs)' %
      (min(vals), max(vals), np.std(vals), max(vals) - min(vals), time.time() - t0))

# ---------- ⑤ 特征重要性集中度 ----------
print('\n=== ⑤ 正式模型增益集中度 ===', flush=True)
gn = m0.feature_importance('gain'); nm = np.array(m0.feature_name())
o = np.argsort(-gn)
print('  Top10:', ', '.join('%s %.1f%%' % (nm[i], gn[i] / gn.sum() * 100) for i in o[:10]))
print('  Top5 占比 %.1f%% | Top10 占比 %.1f%%' %
      (gn[o[:5]].sum() / gn.sum() * 100, gn[o[:10]].sum() / gn.sum() * 100))

pd.DataFrame(rows, columns=['num_leaves', 'min_data', 'oos_ic']).to_csv(
    f'{ROOT}/reports/overfit_grid.csv', index=False)
print('\n已存 reports/overfit_grid.csv')
