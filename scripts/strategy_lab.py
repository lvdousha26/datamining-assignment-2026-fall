# -*- coding: utf-8 -*-
"""自营策略实验 v2：统一用长历史腿（中证红利全收益，2007 起），成本单边 10bp

约束：月频、纯 ETF、不择时、不预测方向。
目的：在 A1（策略③）基础上，找出「更省心 / 更低回撤 / 更简单」的可用变体。
"""
import io, contextlib, os, sys
import numpy as np, pandas as pd

ROOT, BASE = 'D:/quant', r'C:/Users/25097/WorkBuddy/2026-10-30-23-31-35'
COST, RF = 0.0010, 0.02
os.chdir(BASE); sys.path.insert(0, BASE)
with contextlib.redirect_stdout(io.StringIO()):
    import spec3
os.chdir(ROOT)

legs = {k: v.dropna() for k, v in spec3.legs.items()}
a1 = spec3.s3B.dropna()

A = pd.read_pickle(f'{BASE}/assets_m.pkl')
A.index = pd.PeriodIndex(A.index, freq='M')
A = A.loc['2004-01':].ffill()
R = A.pct_change()
BOND = R['国债']


def ann(r):
    r = r.dropna()
    if len(r) < 12: return np.nan, np.nan, np.nan, np.nan
    nav = (1 + r).cumprod()
    a = nav.iloc[-1] ** (12 / len(r)) - 1
    v = r.std() * 12 ** .5
    return a, v, (a - RF) / v, (nav / nav.cummax() - 1).min()


# ---------- 六腿（长历史：A股红利腿用中证红利全收益 div）----------
LEGS = ['mom', 'ma', 'don', 'brk', 'gold', 'div']
L = pd.DataFrame({k: legs[k] for k in LEGS}).dropna()
print('六腿长历史区间：%s ~ %s（%d 月）' % (L.index[0], L.index[-1], len(L)))

# V1 风险平价（36 月波动倒数，单腿≤35%）
vol = L.rolling(36).std().shift(1)
iv = 1 / vol
w = iv.div(iv.sum(axis=1), axis=0)
for _ in range(4):
    if not (w > 0.35).any().any(): break
    w = w.clip(upper=0.35)
    gap = (1 - w.sum(axis=1)).clip(lower=0)
    free = (w < 0.35)
    add = iv.where(free, 0)
    add = add.div(add.sum(axis=1).replace(0, np.nan), axis=0).mul(gap, axis=0)
    w = (w + add.fillna(0)).clip(upper=0.35)
w = w.shift(1).fillna(0)
rp = (w * L).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0) * COST
rp = rp.dropna()

two = 0.5 * L['mom'] + 0.5 * L['div']
three = (L['mom'] + L['div'] + L['gold']) / 3


def with_bond(wb):
    idx = a1.index.intersection(BOND.index)
    return (1 - wb) * a1.reindex(idx) + wb * BOND.reindex(idx)


rv = a1.rolling(6).std() * 12 ** .5
lev = (0.12 / rv).clip(0.5, 1.5).shift(1).fillna(1.0)
a1lev = a1 * lev - (lev - 1) * RF / 12

res = {
    'A1 策略③（基线）': a1,
    'V1 六腿风险平价': rp,
    'V2 两腿：纳指趋势+红利': two,
    'V3 三腿：+黄金': three,
    'V4a A1+国债15%': with_bond(0.15),
    'V4b A1+国债30%': with_bond(0.30),
    'V5 A1 目标波动12%': a1lev,
}

S = lambda v: v[v.index >= pd.Period('2007-02', 'M')]
print('\n' + '=' * 74)
print('%-28s %8s %8s %7s %8s' % ('策略', '年化', '波动', '夏普', '回撤'))
print('=' * 74)
for k, v in res.items():
    a, vv, sh, dd = ann(S(v))
    print('  %-26s %+7.2f%% %7.2f%% %7.2f %7.1f%%' % (k, a * 100, vv * 100, sh, dd * 100))

print('\n' + '=' * 74)
print('分半检验')
print('=' * 74)
print('%-28s %10s %10s %10s' % ('策略', '07-16', '17-26', '后-前'))
for k, v in res.items():
    x = S(v)
    _, _, s1, _ = ann(x[(x.index <= pd.Period('2016-12', 'M'))])
    _, _, s2, _ = ann(x[(x.index >= pd.Period('2017-01', 'M'))])
    print('  %-26s %10.2f %10.2f %+10.2f' % (k, s1, s2, s2 - s1))

tab = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in res.items()}) * 100
print('\n=== 分年收益（%）===')
print(tab.loc[2007:].round(1).to_string())
tab.to_csv(f'{ROOT}/reports/strategy_lab_byyear.csv')
pd.to_pickle(res, f'{ROOT}/data/big/strategy_lab.pkl')
print('\n已存 data/big/strategy_lab.pkl')
