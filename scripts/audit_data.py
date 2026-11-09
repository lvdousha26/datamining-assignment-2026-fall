# -*- coding: utf-8 -*-
"""数据/标签审计：Y2 中的 inf 是否制造了虚假 alpha
背景：fwd5_lag = close[t+6]/close[t+1]-1，当 close[t+1]=0（停牌/数据缺失）会产生 inf。
big_features.py 的 ok 条件只排了 notna()，没排 isfinite()。
"""
import numpy as np, pandas as pd
from scipy import stats as sps

ROOT = r'D:/quant'
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates']); syms = np.array([str(s) for s in z['symbols']])
FEATS = list(z['feats']); FI = {f: i for i, f in enumerate(FEATS)}
T, N, F = X.shape
pr = np.load(f'{ROOT}/data/big/preds.npz')

valid = M & ~np.isnan(Y)
bad = M & np.isinf(Y)
print('=== ① inf 规模与分布 ===')
print('M 内有效样本 %d | 其中 inf %d (%.2f%%)' %
      (valid.sum(), bad.sum(), bad.sum() / valid.sum() * 100))
tb, ib = np.where(bad)
print('按年：', {int(y): int(v) for y, v in
                 pd.Series(dates[tb].year).value_counts().sort_index().items()})
print('涉及股票 %d 只 | 涉及交易日 %d 天' % (len(np.unique(ib)), len(np.unique(tb))))
perday = pd.Series(tb).value_counts()
print('单日 inf 最多 %d 只（%s，当日有效 %d 只）' %
      (perday.max(), dates[perday.index[0]].date(), valid[perday.index[0]].sum()))

print('\n=== ② inf 样本画像（是否为低流动性小票）===')
tv, iv = np.where(valid)
fin = np.isfinite(Y[tv, iv])
for nm, m in [('正常样本', fin), ('inf 样本', ~fin)]:
    xx = X[tv[m], iv[m], :]
    print('  %-8s n=%8d  size %.2f  illiq20 %.2f  turn20 %.2f  amp20 %.2f  volratio %.2f' %
          (nm, m.sum(), xx[:, FI['size']].mean(), xx[:, FI['illiq20']].mean(),
           xx[:, FI['turn20']].mean(), xx[:, FI['amp20']].mean(),
           xx[:, FI['volratio']].mean()))

def ic_of(score, tt, ii, keep):
    d = pd.DataFrame({'d': tt[keep], 'p': score[keep], 'y': Y[tt[keep], ii[keep]]})
    s = d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    return s.mean(), s.mean() / s.std() * np.sqrt(len(s) / 252) if len(s) else np.nan

print('\n=== ③ 单因子 IC：含 inf vs 剔除 inf（全样本）===')
print('  %-14s %10s %10s %10s' % ('因子', '含inf', '剔除inf', '变化'))
for f in ['illiq20', 'illiq_ratio', 'size', 'mf20', 'turn_z', 'amt_ratio',
          'turn20', 'ret5', 'pvcorr20', 'std20']:
    k = FI[f]; s = X[tv, iv, k]
    a, _ = ic_of(s, tv, iv, np.ones(len(tv), bool))
    b, _ = ic_of(s, tv, iv, fin)
    print('  %-14s %+10.4f %+10.4f %+10.4f' % (f, a, b, b - a))

print('\n=== ④ 模型评估：含 inf vs 剔除 inf（测试集 2024+）===')
tt, ii = pr['tt_te'], pr['ii_te']
keep = np.isfinite(Y[tt, ii])
print('  测试集 %d 行 | inf %d 行 (%.2f%%)' % (len(tt), (~keep).sum(), (~keep).mean() * 100))
for nm, key in [('因子合成', 'fac'), ('LGBM', 'p_lgb'), ('TF', 'p_tf'), ('Stacking', 'p_st')]:
    s = pr[key + '_te']
    a, _ = ic_of(s, tt, ii, np.ones(len(tt), bool))
    b, _ = ic_of(s, tt, ii, keep)
    print('  %-10s IC 含inf %+.4f → 剔除 %+.4f（%+.4f）' % (nm, a, b, b - a))

def longshort(s, tt, ii, keep, q=.1, cost=.002):
    d = pd.DataFrame({'t': tt[keep], 'i': ii[keep], 'p': s[keep], 'y': Y[tt[keep], ii[keep]]})
    rec = []
    for t, g in d.groupby('t'):
        if len(g) < 100: continue
        n = max(int(len(g) * q), 10); gs = g.sort_values('p')
        top = gs.tail(n)['y'].mean(); bot = gs.head(n)['y'].mean()
        rest = g[~g.index.isin(gs.head(n).index)]['y'].mean() - g['y'].mean()
        rec.append((top - bot, rest))
    r = pd.DataFrame(rec, columns=['ls', 'ex']).iloc[::5]   # 5 日标签 → 非重叠取样
    out = []
    for c, cst in [('ls', cost * 2), ('ex', cost * .1)]:
        v = r[c] - cst
        ann = (1 + v).prod() ** (50.4 / len(v)) - 1; vol = v.std() * np.sqrt(50.4)
        out.append((ann, (ann - .02) / vol))
    return out

print('\n  多空(20bp/边) 与 剔除垫底10%超额：')
for nm, key in [('LGBM', 'p_lgb'), ('TF', 'p_tf'), ('Stacking', 'p_st')]:
    s = pr[key + '_te']
    (la, lsh), (ea, esh) = longshort(s, tt, ii, np.ones(len(tt), bool))
    (lb, lsh2), (eb, esh2) = longshort(s, tt, ii, keep)
    print('  %-9s 多空 %6.2f%%/%.2f → %6.2f%%/%.2f | 负向筛选 %5.2f%%/%.2f → %5.2f%%/%.2f' %
          (nm, la * 100, lsh, lb * 100, lsh2, ea * 100, esh, eb * 100, esh2))

print('\n=== ⑤ 十分位结构（Stacking，剔除 inf）===')
s = pr['p_st_te']
d = pd.DataFrame({'t': tt[keep], 'p': s[keep], 'y': Y[tt[keep], ii[keep]]})
d['q'] = d.groupby('t')['p'].transform(
    lambda v: pd.qcut(v.rank(method='first'), 10, labels=False))
g = d.groupby('q')['y'].mean() * 100
print('  ', ' '.join('D%d %+.3f%%' % (i + 1, v) for i, v in g.items()))
