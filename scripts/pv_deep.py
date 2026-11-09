# -*- coding: utf-8 -*-
"""量价因子专项深查
1) 逐年 IC 稳定性（2016-2026）——是常年有效还是只有近两年
2) 量价块内部相关性 / PCA 独立维度
3) 可交易性：剔除信号日涨跌停（|ret1|>=9.5%）后 IC 是否还在
4) 垫底档（D1）是什么股票：换手、成交额扩张、波动、市值的画像
"""
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy import stats as sps

ROOT = r'D:/quant'
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates']); FEATS = list(z['feats']); N = X.shape[1]
yr = dates.year
FI = {f: i for i, f in enumerate(FEATS)}

PV = ['mf20', 'amt_ratio', 'illiq_ratio', 'range_ratio', 'turn_z', 'pvcorr20',
      'pvcorr60', 'wvma20', 'vwap_dev', 'clv', 'ushadow', 'lshadow']

def samp(md, mask2=None):
    mm = M & md[:, None] & ~np.isnan(Y)
    if mask2 is not None: mm = mm & mask2
    tt, ii = np.where(mm); k = tt >= 40
    return tt[k], ii[k]

def ic_by_day(pred, tt, ii):
    d = pd.DataFrame({'d': tt, 'p': pred, 'y': Y[tt, ii]})
    s = d.groupby('d').apply(lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    return s

# ---------- 1) 逐年 IC ----------
print('=== 1. 量价因子逐年 IC（截面 Rank IC）===')
print('%-13s' % '因子' + ''.join('%7d' % y for y in range(2016, 2027)))
for f in PV:
    row = []
    for y in range(2016, 2027):
        tt, ii = samp(yr == y)
        if len(tt) == 0: row.append(np.nan); continue
        v = X[tt, ii, FI[f]]
        d = pd.DataFrame({'d': tt, 'p': v, 'y': Y[tt, ii]})
        s = d.groupby('d').apply(lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
        row.append(s.mean())
    print('%-13s' % f + ''.join('%+7.3f' % v if np.isfinite(v) else '      .' for v in row))

# ---------- 2) 相关性与 PCA ----------
print('\n=== 2. 量价块内部相关性（2024+ 样本，逐日截面标准化后合并）===')
tt, ii = samp(yr >= 2024)
sub = X[tt, ii, :][:, [FI[f] for f in PV]]
C = np.corrcoef(sub.T)
print('%-13s' % '' + ''.join('%6s' % f[:5] for f in PV))
for i, f in enumerate(PV):
    print('%-13s' % f + ''.join('%+6.2f' % C[i, j] for j in range(len(PV))))
ev = np.linalg.eigvalsh(np.corrcoef(sub.T))[::-1]
ev = np.clip(ev, 0, None)
print('特征值占比:', np.round(ev / ev.sum() * 100, 1))
print('累计解释 80%% 所需维度: %d / %d' % (int(np.searchsorted(np.cumsum(ev / ev.sum()), .8) + 1), len(PV)))

# ---------- 3) 可交易性：剔除涨跌停 ----------
print('\n=== 3. 剔除信号日涨跌停（|ret1|>=9.5%）后的模型 IC ===')
m = lgb.Booster(model_file=f'{ROOT}/checkpoints/lgbm_big.txt')
r1 = X[:, :, FI['ret1']]
limit = np.abs(r1) >= 0.095
for nm, md in [('2023', yr == 2023), ('2024+', yr >= 2024)]:
    a, b = samp(md); p = m.predict(X[a, b, :])
    ic0 = ic_by_day(p, a, b)
    keep = ~limit[a, b]
    ic1 = ic_by_day(p[keep], a[keep], b[keep])
    print('  %-6s 全部 IC %+.4f (n=%d) | 剔除涨跌停 IC %+.4f (n=%d) | 剔除比例 %.1f%%' % (
        nm, ic0.mean(), len(a), ic1.mean(), keep.sum(), (1 - keep.mean()) * 100))

# ---------- 4) D1 画像 ----------
print('\n=== 4. 垫底档 D1 vs 其余（2024+，模型 Stacking）===')
p = np.load(f'{ROOT}/data/big/preds.npz')
tt, ii, pred = p['tt_te'], p['ii_te'], p['p_st_te']
d = pd.DataFrame({'t': tt, 'i': ii, 'p': pred, 'y': Y[tt, ii]})
d['q'] = d.groupby('t')['p'].transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False))
rows = {}
for f in ['turn_z', 'amt_ratio', 'volratio', 'std20', 'illiq20', 'size', 'ret5', 'mf20', 'pvcorr20', 'range_ratio']:
    rows[f] = d.groupby('q').apply(lambda g, f=f: X[g['t'].values, g['i'].values, FI[f]].mean())
prof = pd.DataFrame(rows).T
print(prof.round(2).to_string())
print('\n  D1 的五日收益: %.2f%% | D10: %.2f%% | 差值 %.2f%%' % (
    d[d['q'] == 0]['y'].mean() * 100, d[d['q'] == 9]['y'].mean() * 100,
    (d[d['q'] == 9]['y'].mean() - d[d['q'] == 0]['y'].mean()) * 100))
