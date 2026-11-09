# -*- coding: utf-8 -*-
"""全市场结果的压力测试：小市值中性化、流动性分层、成本敏感性、纯多头
目的：31% 多空年化到底能不能落地
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'D:/quant'
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, M = z['X'], z['M']
Y = z['Y2']
dates = pd.to_datetime(z['dates']); FEATS = list(z['feats'])
p = np.load(f'{ROOT}/data/big/preds.npz')
SZ, AMT = FEATS.index('size'), FEATS.index('sz_raw') if 'sz_raw' in FEATS else None
print('因子:', FEATS)

MDLS = [('因子合成', 'fac'), ('LGBM', 'p_lgb'), ('TF', 'p_tf'), ('Stacking', 'p_st')]

def frame(key):
    tt, ii = p[f'tt_te'], p['ii_te']
    return pd.DataFrame({'t': tt, 'i': ii, 'p': p[f'{key}_te'], 'y': Y[tt, ii],
                         'sz': X[tt, ii, SZ]})

def perf(r, tag, cost=0.0):
    r = r.iloc[::5] - cost
    ann = (1 + r).prod() ** (50.4 / len(r)) - 1
    vol = r.std() * np.sqrt(50.4)
    print('  %-30s 年化 %7.2f%% 波动 %5.2f%% 夏普 %5.2f' % (tag, ann * 100, vol * 100, (ann - .02) / vol), flush=True)
    return ann, (ann - .02) / vol, (1 + r).cumprod()

def ls_daily(d, q=0.1):
    rec = []
    for t, g in d.groupby('t'):
        if len(g) < 100: continue
        n = max(int(len(g) * q), 10); gs = g.sort_values('p')
        rec.append((dates[t], gs.tail(n)['y'].mean() - gs.head(n)['y'].mean()))
    return pd.Series(dict(rec)).sort_index()

def lo_daily(d, q=0.1):
    rec = []
    for t, g in d.groupby('t'):
        if len(g) < 100: continue
        n = max(int(len(g) * q), 10); gs = g.sort_values('p')
        rec.append((dates[t], gs.tail(n)['y'].mean() - g['y'].mean()))
    return pd.Series(dict(rec)).sort_index()

print('\n=== 1. 基准：多空（费 20bp/边 = 单期 4×20bp）===')
curves = {}
for nm, k in MDLS:
    d = frame(k)
    curves[nm] = perf(ls_daily(d), nm + ' 多空 20bp', cost=0.008)

print('\n=== 2. 成本敏感性（50bp/边）===')
for nm, k in MDLS:
    perf(ls_daily(frame(k)), nm + ' 多空 50bp', cost=0.02)

print('\n=== 3. 流动性分层：只在成交额前 50% 的股票里做多空 ===')
for nm, k in MDLS:
    d = frame(k)
    thr = d.groupby('t')['sz'].transform('median')
    perf(ls_daily(d[d['sz'] >= thr]), nm + ' 高流动性半区', cost=0.008)
print('  —— 只在后 50%（小票）')
for nm, k in MDLS:
    d = frame(k)
    thr = d.groupby('t')['sz'].transform('median')
    perf(ls_daily(d[d['sz'] < thr]), nm + ' 低流动性半区', cost=0.008)

print('\n=== 4. 小市值中性化（逐日对 size 回归取残差）===')
for nm, k in MDLS:
    d = frame(k).copy()
    out = np.empty(len(d))
    for t, g in d.groupby('t'):
        A = np.column_stack([np.ones(len(g)), g['sz'].values])
        b, *_ = np.linalg.lstsq(A, g['p'].values, rcond=None)
        out[d.index.get_indexer(g.index)] = g['p'].values - A @ b
    d['p'] = out
    perf(ls_daily(d), nm + ' 去小市值', cost=0.008)

print('\n=== 5. 纯多头（top10%，超额=相对当日全市场等权）===')
for nm, k in MDLS:
    perf(lo_daily(frame(k)), nm + ' 多头超额 20bp', cost=0.004)

# 图
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.7))
for nm, k in MDLS:
    nav = curves[nm][2]
    ax[0].plot(nav.index, nav.values, lw=1.7, label='%s 夏普%.2f' % (nm, curves[nm][1]))
ax[0].axhline(1, color='#2c3e50', lw=.8, ls='--'); ax[0].legend(frameon=False, fontsize=8.5)
ax[0].grid(alpha=.25); ax[0].set_title('全市场多空净值（2024+，费 20bp/边）', fontsize=10.5)

# 流动性分层柱状
lab, hi, lo = [], [], []
for nm, k in MDLS:
    d = frame(k); thr = d.groupby('t')['sz'].transform('median')
    lab.append(nm)
    hi.append(perf(ls_daily(d[d['sz'] >= thr]), '', cost=0.008)[1])
    lo.append(perf(ls_daily(d[d['sz'] < thr]), '', cost=0.008)[1])
x = np.arange(len(lab))
ax[1].bar(x - .18, hi, .36, label='高流动性半区', color='#2980b9')
ax[1].bar(x + .18, lo, .36, label='低流动性半区', color='#c0392b')
ax[1].axhline(0, color='#2c3e50', lw=.8); ax[1].set_xticks(x); ax[1].set_xticklabels(lab)
ax[1].set_ylabel('多空夏普'); ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=.25, axis='y')
ax[1].set_title('收益来自流动性好的还是差的一半？', fontsize=10.5)
plt.tight_layout(); plt.savefig(f'{ROOT}/reports/big_eval.png', dpi=135, bbox_inches='tight')
print('\n图已保存 reports/big_eval.png')
