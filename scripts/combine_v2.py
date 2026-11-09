# -*- coding: utf-8 -*-
"""A1 × A2 组合（v2，绝对收益口径修正版）

旧版错误：A2 绝对 = (1+A2超额)×(1+沪深300)−1。
但 Y2 的基准是「全市场等权」而非沪深300，且 A2 的超额是在 5 日再平衡口径下测的。
本版直接用逐日模拟出的 A2 绝对月收益，与 A1（spec3 的真实月收益）对齐。
"""
import io, contextlib, os, sys
import numpy as np, pandas as pd

ROOT, BASE = 'D:/quant', r'C:/Users/25097/WorkBuddy/2026-10-30-23-31-35'
_c = os.getcwd(); os.chdir(BASE)
with contextlib.redirect_stdout(io.StringIO()):
    sys.path.insert(0, BASE); import spec3
os.chdir(_c)

a1 = spec3.s3B.dropna()
a2 = pd.read_pickle(f'{ROOT}/data/big/a2_monthly.pkl')
ew = pd.read_pickle(f'{ROOT}/data/big/ew60_monthly.pkl')

j = pd.concat([a1.rename('A1'), a2.rename('A2'), ew.rename('EW')], axis=1).dropna()
print('对齐 %d 个月：%s ~ %s' % (len(j), j.index[0], j.index[-1]))
print('月收益相关：A1-A2 %.3f | A1-EW %.3f | A2-EW %.3f'
      % (j.A1.corr(j.A2), j.A1.corr(j.EW), j.A2.corr(j.EW)))


def st(x, nm, show=True):
    a = (1 + x).prod() ** (12 / len(x)) - 1
    v = x.std() * 12 ** .5
    nav = (1 + x).cumprod(); dd = (nav / nav.cummax() - 1).min()
    sh = (a - 0.02) / v
    if show:
        print('  %-22s 年化 %+7.2f%% | 波动 %6.2f%% | 夏普 %5.2f | 回撤 %6.1f%%'
              % (nm, a * 100, v * 100, sh, dd * 100))
    return a, v, sh, dd


print('\n=== 三条腿单独 ===')
for c in ['A1', 'A2', 'EW']:
    st(j[c], c)

# ---------- ① 直接加权 ----------
print('\n=== ① 直接加权（月频再平衡）===')
print('%8s %9s %9s %8s %9s' % ('w_A2', '年化', '波动', '夏普', '回撤'))
best = (0, -9)
for w in [0, .1, .2, .3, .4, .5, .6, .8, 1.0]:
    c = (1 - w) * j.A1 + w * j.A2
    a, v, sh, dd = st(c, '', False)
    print('%7.0f%% %+8.2f%% %8.2f%% %8.2f %8.1f%%' % (w * 100, a * 100, v * 100, sh, dd * 100))
    if sh > best[1]: best = (w, sh)
print('→ 夏普最优 w_A2 = %.0f%%（夏普 %.2f）' % (best[0] * 100, best[1]))

# ---------- ② A2 加波动目标后再加权 ----------
print('\n=== ② A2 先加波动目标（6 月实现波动，clip 0.2~1.0），再与 A1 加权 ===')
rv = j.A2.rolling(6, min_periods=4).std() * 12 ** .5
for tgt in [0.08, 0.10, 0.12]:
    w = (tgt / rv).clip(0.2, 1.0).shift(1).fillna(0.5)
    a2v = w * j.A2 + (1 - w) * (0.02 / 12)      # 空余仓位放现金（2% 年化）
    a, v, sh, dd = st(a2v, 'A2 波动目标 %.0f%%' % (tgt * 100))
    print('      平均仓位 %.0f%%（区间 %.0f%%~%.0f%%）' % (w.mean() * 100, w.min() * 100, w.max() * 100))

w = (0.10 / rv).clip(0.2, 1.0).shift(1).fillna(0.5)
a2v = w * j.A2 + (1 - w) * (0.02 / 12)
print('\n%8s %9s %9s %8s %9s' % ('w_A2v', '年化', '波动', '夏普', '回撤'))
best2 = (0, -9)
for ww in [0, .1, .2, .3, .4, .5, .6, .8, 1.0]:
    c = (1 - ww) * j.A1 + ww * a2v
    a, v, sh, dd = st(c, '', False)
    print('%7.0f%% %+8.2f%% %8.2f%% %8.2f %8.1f%%' % (ww * 100, a * 100, v * 100, sh, dd * 100))
    if sh > best2[1]: best2 = (ww, sh)
print('→ 夏普最优 w_A2v = %.0f%%（夏普 %.2f）' % (best2[0] * 100, best2[1]))

# ---------- ③ 分年 ----------
print('\n=== ③ 分年净收益（%）===')
comb = 0.7 * j.A1 + 0.3 * j.A2
combv = 0.7 * j.A1 + 0.3 * a2v
rows = []
for y in sorted({p.year for p in j.index}):
    m = np.array([p.year == y for p in j.index])
    rows.append((y, ((1 + j.A1[m]).prod() - 1) * 100, ((1 + j.A2[m]).prod() - 1) * 100,
                 ((1 + comb[m]).prod() - 1) * 100, ((1 + combv[m]).prod() - 1) * 100))
by = pd.DataFrame(rows, columns=['year', 'A1', 'A2', '70/30', '70/30+波动目标'])
print(by.round(2).to_string(index=False))
by.to_csv(f'{ROOT}/reports/A1A2_v2_byyear.csv', index=False)

pd.to_pickle(pd.DataFrame({'A1': j.A1, 'A2': j.A2, 'A2v': a2v, 'EW': j.EW}),
             f'{ROOT}/data/big/A1A2_v2_monthly.pkl')
print('\n已存 data/big/A1A2_v2_monthly.pkl 与 reports/A1A2_v2_byyear.csv')
