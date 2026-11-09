# -*- coding: utf-8 -*-
"""多资产轮动 v2：行情好吃 beta，行情差避险
两层设计：
  层1 风险开关（breadth）：风险资产中在 10 月均线上方的比例 → 决定风险预算
  层2 资产选择：风险资产内部按 9-1 风险调整动量选 top N；防御资产按动量选 1-2
变体：①二分层（硬切换）②连续风险预算 ③波动率目标 ④纯动量 baseline
数据：人民币口径（美元资产×汇率），月频，2007-2026
"""
import io, contextlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DF = r'D:/quant'
BASE = r'C:/Users/25097/WorkBuddy/2026-10-30-23-31-35'
import os
_cwd = os.getcwd(); os.chdir(BASE)
with contextlib.redirect_stdout(io.StringIO()):
    import sys; sys.path.insert(0, BASE)
    import fx_compare as F
os.chdir(_cwd)


import json
def sina_abs(fname):
    d = json.load(open(os.path.join(BASE, fname)))
    t = pd.DataFrame([(x['day'], float(x['close'])) for x in d], columns=['d', 'c'])
    return t.drop_duplicates('d').set_index(pd.to_datetime(t['d'])).sort_index()['c']

def pm(px):
    r = (1 + px.pct_change()).resample('ME').prod() - 1
    r.index = r.index.to_period('M'); return r

RISK = {'沪深300': pm(sina_abs('sina_000300.json')),
        '中证500': pm(pd.read_pickle(BASE+'/idx_000905.pkl')),
        '中证1000': pm(pd.read_pickle(BASE+'/idx_000852.pkl')),
        '红利': F.legs['div'],
        '纳指': pm(F.NDX_R),
        'BRK': pm(F.BRK_R)}
DEF = {'国债': pm(pd.read_pickle(BASE+'/idx_000012.pkl')),
       '黄金': pm(F.GC_R)}
PX = {k: (1+v).cumprod() for k, v in {**RISK, **DEF}.items()}
R = pd.DataFrame({**RISK, **DEF})
R = R.loc['2007-01':].dropna(how='all')
print('资产 %d 个, %s ~ %s (%d月)' % (R.shape[1], R.index[0], R.index[-1], len(R)))

# ---- 信号 ----
logR = np.log1p(R)
MOM = np.expm1(logR.shift(1).rolling(8).sum()) / (R.rolling(9).std()*12**.5 + 1e-6)  # 9-1 RA动量
MA10 = pd.DataFrame({k: (PX[k] > PX[k].rolling(10).mean()).astype(float) for k in PX}).reindex(R.index)
BREADTH = MA10[list(RISK)].mean(axis=1)                     # 风险资产广度 0-1
VOL = R[list(RISK)].mean(axis=1).rolling(6).std()*12**.5     # 组合波动
VOLQ = VOL.rolling(60).apply(lambda s: (s.iloc[-1] >= s).mean())   # 波动率历史分位

COST = 10e-4
def backtest(weights_fn, tag):
    prev = pd.Series(0.0, index=['cash'])
    pr = pd.Series(dtype=float); tos = []
    for i in range(len(R.index) - 1):
        t, nxt = R.index[i], R.index[i+1]
        if t < pd.Period('2007-01'): continue
        w = weights_fn(t, i)
        w = w[w > 0]
        cash_w = max(0, 1 - w.sum())
        r = (w * R.loc[nxt, w.index]).sum() + cash_w * (0.02/12)
        allw = pd.concat([w, pd.Series({'cash': max(0, 1-w.sum())})])
        idx = allw.index.union(prev.index)
        turn = (allw.reindex(idx, fill_value=0) - prev.reindex(idx, fill_value=0)).abs().sum() / 2
        pr.loc[nxt] = r - turn * COST
        tos.append(turn); prev = allw
    return pr, np.mean(tos)

def st(r, tag, to=None):
    a = (1+r).prod()**(12/len(r))-1; v = r.std()*12**.5
    nav = (1+r).cumprod(); dd = (nav/nav.cummax()-1).min()
    print('  %-26s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%% Calmar %.2f%s' % (
        tag, a*100, v*100, (a-.02)/v, dd*100, abs(a/dd), ('  换手%.0f%%/月' % (to*100)) if to else ''))
    return (a-.02)/v, nav

def top_risk(t, n=2, floor=0.0):
    m = MOM.loc[t, list(RISK)].dropna()
    m = m[m > floor].nlargest(n)
    return m / m.sum() if len(m) else pd.Series(dtype=float)

def def_mix(t):
    m = MOM.loc[t, list(DEF)].dropna()
    if len(m) == 0: return pd.Series({'国债': 1.0})
    return m.nlargest(1).clip(lower=1e-9) / m.nlargest(1).clip(lower=1e-9).sum()

# ---- 变体 ----
def v0(t, i):     # baseline: 纯动量 top2 + 绝对动量过滤，其余现金
    return top_risk(t, 2, floor=0.0)
def v1(t, i):     # 二分层：广度≥50%进攻，否则全防御
    if BREADTH.loc[t] >= 0.5:
        return top_risk(t, 2, floor=0.0) * 1.0
    return def_mix(t) * 1.0
def v2(t, i):     # 连续风险预算：w_risk = breadth（防御按动量补足）
    b = np.clip(BREADTH.loc[t], 0, 1)
    w = top_risk(t, 3, floor=0.0) * b
    if b < 1: w = pd.concat([w, def_mix(t) * (1-b)])
    return w
ALL = list(RISK) + list(DEF)
def v4(t, i):     # 统一池：全部资产一起排动量，top2（对照早期版本）
    m = MOM.loc[t, ALL].dropna()
    m = m[m > 0].nlargest(2)
    return m / m.sum() if len(m) else pd.Series(dtype=float)
def v3(t, i):     # 波动率分位控制：波动分位高 → 减仓
    m = top_risk(t, 2, floor=0.0)
    if len(m) == 0: return def_mix(t) * 0.5
    w = np.clip(0.20 / (VOL.loc[t] + 1e-9), 0.2, 1.0)     # 目标20%波动
    return pd.concat([m * w, def_mix(t) * (1-w)]) if w < 1 else m * w

print('\n=== 变体对比（2008-01 ~ %s）===' % R.index[-1])
res = {}
for fn, nm in [(v4, '⓪统一池top2(早期版本)'), (v0, '①纯动量top2(风险池)'), (v1, '②二分层(广度>=50%)'),
               (v2, '③连续风险预算(breadth)'), (v3, '④波动率目标20%')]:
    r, to = backtest(fn, nm)
    res[nm] = st(r, nm, to)[1]

# 基准 & 策略③
for nm, s in [('沪深300 买入持有', pm(sina_abs('sina_000300.json'))),
              ('60/40(300+国债)', 0.6*R['沪深300'] + 0.4*R['国债'])]:
    st(s.loc['2007-01':].dropna(), nm)
os.chdir(BASE)
with contextlib.redirect_stdout(io.StringIO()):
    import spec3
os.chdir(_cwd)
st(spec3.s3B.loc['2007-01':].dropna(), '策略③(参照)')

# 图
fig, ax = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[3, 1.4])
for nm, c, lw in [(list(res)[1], '#c0392b', 2.2), (list(res)[0], '#2980b9', 1.3), (list(res)[3], '#27ae60', 1.2)]:
    ax[0].plot(res[nm].index.to_timestamp('M'), res[nm].values, lw=lw, color=c, label=nm)
s3 = (1+spec3.s3B.loc['2007-01':]).cumprod()
ax[0].plot(s3.index.to_timestamp('M'), s3.values, lw=1.1, color='#7f8c8d', ls='--', label='策略③(参照)')
ax[0].set_yscale('log'); ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=.25, which='both')
ax[0].set_title('多资产轮动：行情好吃 beta / 行情差避险（费后，人民币口径）', fontsize=11)
ax[1].plot(BREADTH.index.to_timestamp('M'), BREADTH.values*100, color='#8e44ad', lw=1.1)
ax[1].axhline(50, color='#7f8c8d', ls=':', lw=1)
ax[1].set_ylabel('风险资产广度 %'); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig(f'{DF}/reports/regime_rotation.png', dpi=135, bbox_inches='tight')
print('\n图已保存 reports/regime_rotation.png')
print('\n当前广度 %.0f%%（%d/%d 个风险资产在10月均线上）| 当前波动 %.0f%%' % (
    BREADTH.iloc[-1]*100, MA10[list(RISK)].iloc[-1].sum(), len(RISK), VOL.iloc[-1]*100))
print('风险资产动量排名:', MOM.iloc[-1][list(RISK)].sort_values(ascending=False).round(2).to_dict())
