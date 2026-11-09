# -*- coding: utf-8 -*-
"""行业补齐 + 行业中性化检验
1) 抓申万一级 31 个行业成分 → symbol 映射
2) D1 的行业分布 vs 全市场基准
3) 行业中性化：预测与收益都在行业内去均值后，IC 还剩多少
4) 行业中性化后的负向筛选（每个行业内剔除垫底 10%）
"""
import time, numpy as np, pandas as pd, akshare as ak
from scipy import stats as sps

ROOT = r'D:/quant'
MAPF = f'{ROOT}/data/big/ind_map.parquet'

def fetch_map():
    inds = ak.sw_index_first_info()
    print('申万一级 %d 个' % len(inds), flush=True)
    rows = []
    for _, r in inds.iterrows():
        code, name = str(r['行业代码']).split('.')[0], r['行业名称']
        for attempt in range(3):
            try:
                d = ak.index_component_sw(symbol=code)
                if len(d):
                    d = d[['证券代码', '证券名称']].copy()
                    d['industry'] = name
                    rows.append(d)
                break
            except Exception:
                time.sleep(0.6)
        time.sleep(0.15)
    m = pd.concat(rows, ignore_index=True)
    m.columns = ['symbol', 'name', 'industry']
    m['symbol'] = m['symbol'].astype(str).str.zfill(6)
    m = m.drop_duplicates('symbol')
    m.to_parquet(MAPF, index=False)
    print('映射 %d 只' % len(m), flush=True)
    return m

import os
if os.path.exists(MAPF):
    m = pd.read_parquet(MAPF)
    print('已有行业映射 %d 只' % len(m), flush=True)
else:
    m = fetch_map()

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates']); syms = np.array([str(s) for s in z['symbols']])
FEATS = list(z['feats']); FI = {f: i for i, f in enumerate(FEATS)}
ind_of = dict(zip(m['symbol'], m['industry']))
IND = np.array([ind_of.get(s, '未分类') for s in syms])
print('覆盖：%d/%d 只有行业（%.1f%%）' % ((IND != '未分类').sum(), len(syms), (IND != '未分类').mean() * 100), flush=True)
vc = pd.Series(IND).value_counts()
print('最大行业 %s %d 只；未分类 %d 只' % (vc.index[0], vc.iloc[0], (IND == '未分类').sum()), flush=True)

p = np.load(f'{ROOT}/data/big/preds.npz')
a, b, pred = p['tt_te'], p['ii_te'], p['p_st_te']
yr = dates.year

def ic(pred, tt, ii, y):
    d = pd.DataFrame({'d': tt, 'p': pred, 'y': y}).dropna()
    s = d.groupby('d').apply(lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    return s.mean()

# ---------- 1) D1 行业分布 ----------
print('\n=== 1. 垫底档 D1 的行业分布（2024+，Stacking）===')
d = pd.DataFrame({'t': a, 'i': b, 'p': pred, 'y': Y[a, b], 'ind': IND[b]})
d['q'] = d.groupby('t')['p'].transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False))
share_all = d['ind'].value_counts(normalize=True)
share_d1 = d[d['q'] == 0]['ind'].value_counts(normalize=True)
cmp = pd.DataFrame({'全市场占比': share_all, 'D1占比': share_d1}).fillna(0)
cmp['超额倍数'] = (cmp['D1占比'] + 1e-9) / (cmp['全市场占比'] + 1e-9)
cmp = cmp.sort_values('超额倍数', ascending=False)
print(cmp.head(8).round(3).to_string())
print('...')
print(cmp.tail(5).round(3).to_string())
print('\n  D1 收益按行业（前后 5）：')
g = d[d['q'] == 0].groupby('ind')['y'].agg(['mean', 'count'])
g = g[g['count'] > 2000].sort_values('mean')
print((g.head(5) * [100, 1]).round(2).to_string())
print((g.tail(5) * [100, 1]).round(2).to_string())

# ---------- 2) 行业中性化 IC ----------
print('\n=== 2. 行业中性化（预测与收益都在 日×行业 内去均值）===')
dd = pd.DataFrame({'d': a, 'p': pred, 'y': Y[a, b], 'ind': IND[b]})
dd['p_n'] = dd['p'] - dd.groupby(['d', 'ind'])['p'].transform('mean')
dd['y_n'] = dd['y'] - dd.groupby(['d', 'ind'])['y'].transform('mean')
print('  原始   IC %+.4f' % ic(dd['p'], dd['d'], np.arange(len(dd)), dd['y']))
print('  中性化 IC %+.4f（行业内选股能力）' % ic(dd['p_n'], dd['d'], np.arange(len(dd)), dd['y_n']))
# 行业层面解释力：行业均值收益能被预测的行业均值解释多少
gi = dd.groupby(['d', 'ind']).agg(p=('p', 'mean'), y=('y', 'mean')).reset_index()
gi['pr'] = gi.groupby('d')['p'].rank(pct=True)
ic_ind = gi.groupby('d').apply(lambda g: sps.spearmanr(g['pr'], g['y']).correlation if len(g) > 5 else np.nan).dropna()
print('  行业层面 IC（预测的行业均值 vs 行业平均收益）%+.4f' % ic_ind.mean())

# ---------- 3) 行业中性化后的负向筛选 ----------
print('\n=== 3. 负向筛选：每个行业内剔除垫底 10%（2024+，单边 20bp）===')
rec = []
for t, g in dd.groupby('d'):
    keep = []
    for ind, gg in g.groupby('ind'):
        if len(gg) < 10:
            continue
        n = max(int(len(gg) * 0.1), 2)
        keep.append(gg.sort_values('p').iloc[n:])   # 去掉分数最低的 n 只
    rest = pd.concat(keep) if keep else g
    rec.append((dates[t], rest['y'].mean() - g['y'].mean()))
r = pd.DataFrame(rec, columns=['d', 'r']).set_index('d')['r'].iloc[::5] - 0.0004
ann = (1 + r).prod() ** (50.4 / len(r)) - 1; vol = r.std() * np.sqrt(50.4)
print('  行业中性负向筛选 年化 %+.2f%% 波动 %.2f%% 夏普 %.2f' % (ann * 100, vol * 100, (ann - .02) / vol))
