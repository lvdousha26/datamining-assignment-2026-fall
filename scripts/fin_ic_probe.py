# -*- coding: utf-8 -*-
"""基本面因子验证：单因子 IC、与量价因子相关性、Ridge 增量

三段：
① 单因子 IC/ICIR（全样本 + 分年）
② 与 42 个量价因子的相关性（逐日截面秩相关的均值）
③ 在 dev 段把「基本面合成分」作为 Ridge 第 4 输入，对比已知 LGBM+TF 的 0.1412 和
   LGBM+TF+量价合成 0.1451
"""
import numpy as np, pandas as pd
from scipy import stats as sps

ROOT = 'D:/quant'
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y2, M, dates, symbols = z['X'], z['Y2'], z['M'], z['dates'], z['symbols']
T, N = X.shape[:2]
yrs = pd.DatetimeIndex(dates).year.values

fp = pd.read_parquet(f'{ROOT}/data/big/fin_panel.parquet')
fp['date'] = pd.to_datetime(fp['date'])
dmap = {d: i for i, d in enumerate(pd.to_datetime(pd.Series(dates).astype(str)))}
smap = {str(s).zfill(6): j for j, s in enumerate(symbols)}
fp = fp[fp.date.isin(dmap) & fp.symbol.isin(smap)].drop_duplicates(['date', 'symbol'])
ti = fp.date.map(dmap).values.astype(np.int32)
si = fp.symbol.map(smap).values.astype(np.int32)
print('对齐 %d 行 / %d 日 / %d 只' % (len(fp), len(np.unique(ti)), len(np.unique(si))), flush=True)

FIN = ['roe_ttm', 'roa_ttm', 'gpa', 'gross_margin', 'ocf_ni', 'accrual', 'debt_ratio',
       'asset_turn', 'roe_stab', 'roe_trend', 'gm_trend', 'rev_yoy', 'profit_yoy',
       'recv_gap', 'inv_gap', 'ocf_yoy', 'ep', 'bp', 'sp', 'ocf_yield', 'fcf_yield']
# 方向：账面上"越大越好"的因子标 +1；越小越好的标 -1
DIR = dict(roe_ttm=1, roa_ttm=1, gpa=1, gross_margin=1, ocf_ni=1, accrual=-1, debt_ratio=-1,
           asset_turn=1, roe_stab=-1, roe_trend=1, gm_trend=1, rev_yoy=1, profit_yoy=1,
           recv_gap=-1, inv_gap=-1, ocf_yoy=1, ep=1, bp=1, sp=1, ocf_yield=1, fcf_yield=1)

# 截面 z（逐日）
Z = {}
for c in FIN:
    v = fp[c].values.astype(np.float64)
    s = pd.Series(v).groupby(ti).transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    Z[c] = s.values * DIR[c]


def daily_ic(pred, mask):
    d = pd.DataFrame({'t': ti, 'p': pred, 'y': Y2[ti, si]})
    d = d[mask & np.isfinite(d.y)]
    out = []
    for t, g in d.groupby('t'):
        if len(g) < 100: continue
        out.append((t, sps.spearmanr(g.p, g.y).correlation))
    return pd.DataFrame(out, columns=['t', 'ic'])


print('\n=== ① 单因子 IC（方向已统一为越大越好）===')
rows = []
for c in FIN:
    m = np.isfinite(Z[c]) & M[ti, si] & np.isfinite(Y2[ti, si])
    if m.sum() < 5000:
        print('%-12s 样本不足' % c); continue
    ic = daily_ic(Z[c], m)
    if len(ic) < 50: continue
    ic['year'] = pd.DatetimeIndex(dates[ic.t]).year
    by = ic.groupby('year')['ic'].mean()
    rows.append(dict(factor=c, n=len(ic), ic=ic.ic.mean(), icir=ic.ic.mean() / ic.ic.std(),
                     pos=(ic.ic > 0).mean(), ic_16_19=by.loc[2016:2019].mean(),
                     ic_20_22=by.loc[2020:2022].mean(), ic_23_25=by.loc[2023:2025].mean(),
                     ic_26=by.loc[2026].mean() if 2026 in by.index else np.nan))
res = pd.DataFrame(rows).sort_values('icir', ascending=False)
print(res.round(3).to_string(index=False))
res.to_csv(f'{ROOT}/reports/fin_ic.csv', index=False)

# ---------- 合成分 ----------
qual = ['roe_ttm', 'roa_ttm', 'gpa', 'ocf_ni', 'accrual', 'debt_ratio', 'roe_stab', 'roe_trend', 'gm_trend']
valu = ['ep', 'bp', 'sp', 'ocf_yield', 'fcf_yield']
Q = np.nanmean(np.column_stack([Z[c] for c in qual]), axis=1)
V = np.nanmean(np.column_stack([Z[c] for c in valu]), axis=1)
Q = pd.Series(Q).groupby(ti).transform(lambda x: (x - x.mean()) / (x.std() + 1e-12)).values
V = pd.Series(V).groupby(ti).transform(lambda x: (x - x.mean()) / (x.std() + 1e-12)).values
COMP = 0.5 * Q + 0.5 * V

for nm, v in [('质量分', Q), ('估值分', V), ('质量+估值', COMP)]:
    m = np.isfinite(v) & M[ti, si] & np.isfinite(Y2[ti, si])
    ic = daily_ic(v, m)
    ic['year'] = pd.DatetimeIndex(dates[ic.t]).year
    by = ic.groupby('year')['ic'].mean()
    print('%-8s IC %+.4f  ICIR %.2f  | 16-19 %+.3f 20-22 %+.3f 23-25 %+.3f 26 %+.3f' %
          (nm, ic.ic.mean(), ic.ic.mean() / ic.ic.std(),
           by.loc[2016:2019].mean(), by.loc[2020:2022].mean(), by.loc[2023:2025].mean(),
           by.loc[2026].mean() if 2026 in by.index else np.nan))

# ---------- ② 与量价因子相关性 ----------
print('\n=== ② 基本面合成分 vs 42 量价因子（秩相关，截面 z 后近似）===')
feats = z['feats'].tolist()
m = np.isfinite(COMP) & M[ti, si]
cc = []
for k, f in enumerate(feats):
    xk = X[ti, si, k]
    ok = m & np.isfinite(xk)
    if ok.sum() < 10000: continue
    r1 = pd.Series(COMP[ok]).rank().values
    r2 = pd.Series(xk[ok]).rank().values
    cc.append((f, np.corrcoef(r1, r2)[0, 1]))
cc = pd.DataFrame(cc, columns=['factor', 'corr']).sort_values('corr', key=lambda s: s.abs(), ascending=False)
print(cc.head(8).round(3).to_string(index=False))
print('平均 |相关| %.3f' % cc['corr'].abs().mean())

# ---------- ③ Ridge 增量（复用已有预测）----------
print('\n=== ③ dev 段 Ridge 增量 ===')
p = np.load(f'{ROOT}/data/big/preds.npz')
tt, ii = p['tt_te'], p['ii_te']
base = np.full((T, N), np.nan)
for c in ['ep', 'bp', 'sp', 'ocf_yield', 'fcf_yield', 'roe_ttm', 'gpa', 'ocf_ni']:
    base[ti, si] = np.where(np.isfinite(fp[c].values), fp[c].values, base[ti, si])
fin_dev = np.full(len(tt), np.nan)
key = pd.Series(ti * N + si)
lookup = pd.Series(COMP).groupby(key).mean()
lookupV = pd.Series(V).groupby(key).mean()
lookupQ = pd.Series(Q).groupby(key).mean()
idx_dev = pd.Series(tt * N + ii)
fin_dev = lookup.reindex(idx_dev).values
val_dev = lookupV.reindex(idx_dev).values
qua_dev = lookupQ.reindex(idx_dev).values


def ridge_fit(A, y, lam):
    n = len(y)
    mu, sd = A[:, :-1].mean(0), A[:, :-1].std(0) + 1e-12
    As = np.column_stack([(A[:, :-1] - mu) / sd, np.ones(n)])
    pen = np.eye(As.shape[1]) * lam * n
    pen[-1, -1] = 0
    return np.linalg.solve(As.T @ As + pen, As.T @ y), mu, sd


def zc(a):
    a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-12)


def eval_pred(pr, name):
    d = pd.DataFrame({'t': tt, 'p': pr, 'y': Y2[tt, ii]})
    s = d.groupby('t').apply(lambda g: sps.spearmanr(g.p, g.y).correlation if len(g) > 30 else np.nan,
                             include_groups=False).dropna()
    print('  %-34s IC %+.4f  ICIR %.2f  (%d 日)' % (name, s.mean(), s.mean() / s.std(), len(s)))


valid = np.isfinite(fin_dev)
print('  基本面分在 dev 段覆盖 %d / %d 行' % (valid.sum(), len(fin_dev)))
pl, pt = p['p_lgb_te'], p['p_tf_te']
pf = p['fac_te']
if valid.mean() > 0.5:
    fv = np.where(valid, fin_dev, np.nanmedian(fin_dev))
    vv = np.where(np.isfinite(val_dev), val_dev, np.nanmedian(val_dev))
    qv = np.where(np.isfinite(qua_dev), qua_dev, np.nanmedian(qua_dev))
    print('  估值分/质量分覆盖 %.1f%% / %.1f%%' % (np.isfinite(val_dev).mean() * 100,
                                                 np.isfinite(qua_dev).mean() * 100))
    for nm, cols in [('LGBM+TF', [pl, pt]),
                     ('LGBM+TF+量价合成', [pl, pt, pf]),
                     ('LGBM+TF+估值分', [pl, pt, vv]),
                     ('LGBM+TF+质量分', [pl, pt, qv]),
                     ('LGBM+TF+基本面合成', [pl, pt, fv]),
                     ('LGBM+TF+量价+估值', [pl, pt, pf, vv]),
                     ('LGBM+TF+量价+基本面', [pl, pt, pf, fv])]:
        A = np.column_stack([zc(c) for c in cols] + [np.ones(len(pl))])
        y = Y2[tt, ii]; yz = (y - y.mean()) / y.std()
        c, mu, sd = ridge_fit(A, yz, 1e-3)
        eval_pred(A @ c, nm + '  coef=' + str(np.round(c[:-1], 3)))
