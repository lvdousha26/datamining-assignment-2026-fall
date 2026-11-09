# -*- coding: utf-8 -*-
"""A2 加债券/黄金做风险预算：把 17.7% 波动压到 8-12%，看夏普能到多少

A2 月度绝对收益来自 combine_A1_A2 的算法（季度超额 + 沪深300）。
对照：纯 A2 / 固定权重 / 波动率目标（动态）
"""
import io, contextlib, os, sys
import numpy as np, pandas as pd

ROOT = 'D:/quant'
BASE = r'C:/Users/25097/WorkBuddy/2026-10-30-23-31-35'

# ---------- 复用 A2 月度序列 ----------
import subprocess
exec(open(f'{ROOT}/scripts/combine_A1_A2.py', encoding='utf-8').read().split('# ---------- A1 ----------')[0])
print('A2 月度 %d | %s ~ %s' % (len(a2_abs), a2_abs.index[0], a2_abs.index[-1]), flush=True)

# ---------- 债券 / 黄金 ----------
bond_px = pd.read_pickle(f'{BASE}/idx_000012.pkl')
bond_m = (1 + bond_px.pct_change()).resample('ME').prod() - 1
bond_m.index = bond_m.index.to_period('M')

os.chdir(BASE)
with contextlib.redirect_stdout(io.StringIO()):
    sys.path.insert(0, BASE)
    import spec3
os.chdir(ROOT)
gold = spec3.legs['gold']          # 月度
gold.index = pd.PeriodIndex(gold.index, freq='M')

j = pd.concat([a2_abs.rename('A2'), bond_m.rename('BOND'), gold.rename('GOLD')], axis=1).dropna()
print('对齐 %d 月 | 相关 A2-债 %.2f | A2-金 %.2f | 债-金 %.2f' %
      (len(j), j.A2.corr(j.BOND), j.A2.corr(j.GOLD), j.BOND.corr(j.GOLD)), flush=True)


def st(x, nm):
    a = (1 + x).prod() ** (12 / len(x)) - 1
    v = x.std() * 12 ** .5
    nav = (1 + x).cumprod(); dd = (nav / nav.cummax() - 1).min()
    print('  %-26s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%%' %
          (nm, a * 100, v * 100, (a - .02) / v, dd * 100))
    return (a - .02) / v, v


print('\n=== ① 固定权重 ===')
res = {}
for nm, w in [('纯 A2', (1, 0, 0)), ('A2 80/债 20', (.8, .2, 0)),
              ('A2 70/债 30', (.7, .3, 0)), ('A2 60/债 40', (.6, .4, 0)),
              ('A2 50/债 50', (.5, .5, 0)),
              ('A2 60/债 25/金 15', (.6, .25, .15)),
              ('A2 50/债 35/金 15', (.5, .35, .15))]:
    c = w[0] * j.A2 + w[1] * j.BOND + w[2] * j.GOLD
    sh, v = st(c, nm)
    res[nm] = (c, sh, v)

print('\n=== ② 波动率目标（过去 6 月实现波动，不足仓位给 债 70%/金 30%）===')
for tgt in [0.08, 0.10, 0.12]:
    rv = j.A2.rolling(6, min_periods=4).std() * 12 ** .5
    w = (tgt / rv).clip(0.2, 1.0).shift(1).fillna(0.5)
    c = w * j.A2 + (1 - w) * (0.7 * j.BOND + 0.3 * j.GOLD)
    sh, v = st(c, '目标波动 %.0f%%' % (tgt * 100))
    res['vol%d' % (tgt * 100)] = (c, sh, v)
    print('     平均仓位 %.0f%%（%.0f%%~%.0f%%）' % (w.mean() * 100, w.min() * 100, w.max() * 100))

print('\n=== ③ 分年（净收益 %）===')
tab = pd.DataFrame({k: v[0].groupby(v[0].index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
                    for k, v in res.items()})
print(tab.round(1).to_string())

print('\n=== ④ 与 A1 比（同段 2016-2026）===')
a1 = spec3.s3B.dropna()
a1 = a1[a1.index.isin(j.index)]
st(a1, 'A1 策略③（同段）')

print('\n结论：夏普从 %.2f（纯 A2）提升到 %.2f（最优固定权重）/ %.2f（最优波动目标）' %
      (res['纯 A2'][1],
       max(v[1] for k, v in res.items() if k.startswith('A2')),
       max(v[1] for k, v in res.items() if k.startswith('vol'))))
tab.round(2).to_csv(f'{ROOT}/reports/A2_riskbudget_byyear.csv')
print('已存 reports/A2_riskbudget_byyear.csv')
