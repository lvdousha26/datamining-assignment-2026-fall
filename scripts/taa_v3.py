# -*- coding: utf-8 -*-
"""TAA v3：ERP 择时 + 精简资产池

相对于 v2（regime_rotation.py）的两处改动：
1. 资产池：6 个风险资产（4 个 A 股宽基，两两相关 0.8+）→ 4 个（沪深300 / 中证1000 / 纳指 / BRK）
   防御仍为国债 + 黄金。
2. 信号：动量广度 → **股债性价比 ERP 的历史分位**（沪深300 PE-TTM 倒数 − 国债收益率）
   ERP 是均值回归信号，月度频率稳定；价格动量在 A 股弱且噪声大（v2 五个变体夏普 0.15-0.25）

变体：
  A  ERP 线性预算 + 风险等权
  B  ERP 线性预算 + 动量 top2
  C  ERP 线性预算 + 波动率目标 20%
  D  v2 最好的那个（动量广度 + 波动率目标）作对照
"""
import io, contextlib, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DF = r'D:/quant'
BASE = r'C:/Users/25097/WorkBuddy/2026-10-30-23-31-35'
_cwd = os.getcwd(); os.chdir(BASE)
with contextlib.redirect_stdout(io.StringIO()):
    sys.path.insert(0, BASE)
    import fx_compare as F
os.chdir(_cwd)


def sina_abs(fname):
    d = json.load(open(os.path.join(BASE, fname)))
    t = pd.DataFrame([(x['day'], float(x['close'])) for x in d], columns=['d', 'c'])
    return t.drop_duplicates('d').set_index(pd.to_datetime(t['d'])).sort_index()['c']


def pm(px):
    r = (1 + px.pct_change()).resample('ME').prod() - 1
    r.index = r.index.to_period('M'); return r


# ---------- 精简后的资产池 ----------
RISK = {'沪深300': pm(sina_abs('sina_000300.json')),
        '中证1000': pm(pd.read_pickle(BASE + '/idx_000852.pkl')),
        '纳指': pm(F.NDX_R),
        'BRK': pm(F.BRK_R)}
DEF = {'国债': pm(pd.read_pickle(BASE + '/idx_000012.pkl')),
       '黄金': pm(F.GC_R)}
R = pd.DataFrame({**RISK, **DEF}).loc['2007-01':].dropna(how='all')
print('资产 %d 个（风险 %d / 防御 %d）, %s ~ %s (%d 月)' %
      (R.shape[1], len(RISK), len(DEF), R.index[0], R.index[-1], len(R)), flush=True)

# 相关性诊断（说明为什么要去掉中证500和红利）
corr = R[list(RISK)].corr()
print('\n风险资产相关性：')
print(corr.round(2).to_string(), flush=True)
print('平均相关 %.2f' % ((corr.values.sum() - len(corr)) / (len(corr) ** 2 - len(corr))), flush=True)

# ---------- ERP ----------
pe = pd.read_csv(f'{DF}/策略_策略3定稿/pe_沪深30.csv', parse_dates=['日期']).set_index('日期')
pe = pe['滚动市盈率'].astype(float).resample('ME').last()
pe.index = pe.index.to_period('M')
bond = pd.read_pickle(f'{DF}/bond_5y.pkl')['国债'].astype(float)
bond = bond.resample('ME').mean()
bond.index = bond.index.to_period('M')
ERP = (100.0 / pe - bond).dropna()
ERP = ERP.reindex(R.index).ffill()
print('\nERP: %s ~ %s | 当前 %.2f%% | 均值 %.2f%%' %
      (ERP.first_valid_index(), ERP.dropna().index[-1], ERP.dropna().iloc[-1], ERP.mean()), flush=True)


def pct_rank(s, win=60):
    return s.rolling(win, min_periods=24).apply(lambda x: (x.iloc[-1] >= x).mean())


ERP_Q = pct_rank(ERP, 60)          # 0-1：越高越便宜
print('ERP 分位（60月滚动）当前 %.0f%%' % (ERP_Q.dropna().iloc[-1] * 100), flush=True)

# ---------- 信号 ----------
PX = {k: (1 + v).cumprod() for k, v in {**RISK, **DEF}.items()}
logR = np.log1p(R)
MOM = np.expm1(logR.shift(1).rolling(8).sum()) / (R.rolling(9).std() * 12 ** .5 + 1e-6)
MA10 = pd.DataFrame({k: (PX[k] > PX[k].rolling(10).mean()).astype(float) for k in PX}).reindex(R.index)
VOL = R[list(RISK)].mean(axis=1).rolling(6).std() * 12 ** .5

COST = 10e-4


def backtest(weights_fn):
    prev = pd.Series(0.0, index=['cash'])
    pr = pd.Series(dtype=float); tos = []
    for i in range(len(R.index) - 1):
        t, nxt = R.index[i], R.index[i + 1]
        w = weights_fn(t, i)
        if w is None or (len(w) == 0 and False):
            w = pd.Series(dtype=float)
        w = w[w > 0]
        cash_w = max(0.0, 1 - w.sum())
        r = (w * R.loc[nxt, w.index]).sum() + cash_w * (0.02 / 12)
        allw = pd.concat([w, pd.Series({'cash': cash_w})])
        idx = allw.index.union(prev.index)
        turn = (allw.reindex(idx, fill_value=0) - prev.reindex(idx, fill_value=0)).abs().sum() / 2
        pr.loc[nxt] = r - turn * COST
        tos.append(turn); prev = allw
    return pr, float(np.mean(tos))


def st(r, tag, to=None):
    r = r.dropna()
    a = (1 + r).prod() ** (12 / len(r)) - 1; v = r.std() * 12 ** .5
    nav = (1 + r).cumprod(); dd = (nav / nav.cummax() - 1).min()
    print('  %-30s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%% Calmar %.2f%s' %
          (tag, a * 100, v * 100, (a - .02) / v, dd * 100, abs(a / dd),
           ('  换手%.0f%%/月' % (to * 100)) if to else ''))
    return nav


def risk_eq(t, n=None):
    m = MOM.loc[t, list(RISK)].dropna()
    m = m[m > 0]
    if len(m) == 0: return pd.Series(dtype=float)
    if n: m = m.nlargest(n)
    return m / m.sum() * 0 + 1.0 / len(m) if False else pd.Series(1.0 / len(m), index=m.index)


def risk_mom(t, n=2):
    m = MOM.loc[t, list(RISK)].dropna()
    m = m[m > 0].nlargest(n)
    return m / m.sum() if len(m) else pd.Series(dtype=float)


def def_mix(t):
    m = MOM.loc[t, list(DEF)].dropna()
    if len(m) == 0: return pd.Series({'国债': 1.0})
    m = m.nlargest(1)
    return m / m.sum()


def budget(t, lo=0.2, hi=1.0):
    q = ERP_Q.get(t, np.nan)
    if not np.isfinite(q): return 0.5
    return float(np.clip(lo + (hi - lo) * q, lo, hi))


def vA(t, i):      # ERP 预算 + 风险等权
    b = budget(t); w = risk_eq(t) * b
    return pd.concat([w, def_mix(t) * (1 - b)]) if b < 1 else w


def vB(t, i):      # ERP 预算 + 动量 top2
    b = budget(t); w = risk_mom(t, 2) * b
    return pd.concat([w, def_mix(t) * (1 - b)]) if b < 1 else w


def vC(t, i):      # ERP 预算 + 波动率目标
    b = budget(t)
    base = risk_mom(t, 2)
    if len(base) == 0: return def_mix(t) * (1 - b)
    vol = VOL.get(t, np.nan)
    vb = float(np.clip(0.20 / (vol + 1e-9), 0.2, 1.0)) if np.isfinite(vol) else 1.0
    w = base * b * vb
    return pd.concat([w, def_mix(t) * (1 - w.sum())]) if w.sum() < 1 else w


# v2 对照：动量广度 + 波动率目标
BREADTH = MA10[list(RISK)].mean(axis=1)


def vD(t, i):
    m = risk_mom(t, 2)
    if len(m) == 0: return def_mix(t) * 0.5
    w = float(np.clip(0.20 / (VOL.loc[t] + 1e-9), 0.2, 1.0))
    return pd.concat([m * w, def_mix(t) * (1 - w)]) if w < 1 else m * w


def vE(t, i):      # 纯 ERP 择时 + 4 风险资产固定等权（不做动量过滤）
    b = budget(t)
    w = pd.Series(1.0 / len(RISK), index=list(RISK)) * b
    return pd.concat([w, def_mix(t) * (1 - b)]) if b < 1 else w


print('\n=== 变体对比（费后，人民币口径）===')
res = {}
for fn, nm in [(vA, 'A ERP预算 + 风险等权'), (vB, 'B ERP预算 + 动量top2'),
               (vC, 'C ERP预算 + 波动率目标'), (vE, 'E 纯ERP + 风险固定等权'),
               (vD, 'D 动量广度+波动率(v2最好)')]:
    r, to = backtest(fn)
    st(r, nm, to)
    res[nm] = r            # 存收益序列（不是净值），组合与画图都依赖它

for nm, s in [('沪深300 买入持有', pm(sina_abs('sina_000300.json')).loc[R.index[0]:]),
              ('60/40(300+国债)', 0.6 * R['沪深300'] + 0.4 * R['国债'])]:
    st(s.dropna(), nm)

os.chdir(BASE)
with contextlib.redirect_stdout(io.StringIO()):
    import spec3
os.chdir(_cwd)
nav3 = st(spec3.s3B.loc['2007-01':].dropna(), '策略③(参照)')

# 相关性：新 TAA 与策略③ 是否互补
rA = res['A ERP预算 + 风险等权']
j = pd.concat([rA.rename('taaA'), spec3.s3B.rename('s3')], axis=1).dropna()
print('\nvA 与策略③ 月收益相关 %.2f | 策略③ 同期夏普 %.2f' %
      (j['taaA'].corr(j['s3']), (j['s3'].mean() * 12) / (j['s3'].std() * 12 ** .5)), flush=True)

# ---------- 与策略③ 的组合 ----------
rE = res['E 纯ERP + 风险固定等权']
jj = pd.concat([rE.rename('e'), spec3.s3B.rename('s3')], axis=1).dropna()
print('\n=== 策略③ + TAA-E 组合（月频再平衡）===')
print('%6s %8s %8s %7s %7s' % ('w_TAA', '年化', '波动', '夏普', '回撤'))
best = (None, -9)
for w in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    c = (1 - w) * jj['s3'] + w * jj['e']
    a = (1 + c).prod() ** (12 / len(c)) - 1
    v = c.std() * 12 ** .5
    nav = (1 + c).cumprod(); dd = (nav / nav.cummax() - 1).min()
    sh = (a - .02) / v
    print('%6.0f%% %7.2f%% %7.2f%% %7.2f %6.1f%%' % (w * 100, a * 100, v * 100, sh, dd * 100))
    if sh > best[1]: best = (w, sh)
print('→ 夏普最优 w_TAA = %.0f%%（夏普 %.2f）' % (best[0] * 100, best[1]))
print('  两者月收益相关 %.2f' % jj['e'].corr(jj['s3']))

# 图
fig, ax = plt.subplots(3, 1, figsize=(11, 9), height_ratios=[3, 1.2, 1.2])
for nm, c, lw in [(list(res)[0], '#c0392b', 2.2), (list(res)[2], '#27ae60', 1.5),
                  (list(res)[3], '#2980b9', 1.2)]:
    ax[0].plot(res[nm].index.to_timestamp('M'), res[nm].cumprod().values, lw=lw, color=c, label=nm)
ax[0].plot(nav3.index.to_timestamp('M'), nav3.values, lw=1.1, color='#7f8c8d', ls='--', label='策略③')
ax[0].set_yscale('log'); ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=.25, which='both')
ax[0].set_title('TAA v3：ERP 择时 + 精简资产池（费后）', fontsize=11)
ax[1].plot(ERP.index.to_timestamp('M'), ERP.values, color='#8e44ad', lw=1.1)
ax[1].set_ylabel('ERP %'); ax[1].grid(alpha=.25)
ax[2].plot(ERP_Q.index.to_timestamp('M'), ERP_Q.values * 100, color='#e67e22', lw=1.1)
ax[2].axhline(50, color='#7f8c8d', ls=':', lw=1); ax[2].set_ylabel('ERP 分位 %'); ax[2].grid(alpha=.25)
plt.tight_layout(); plt.savefig(f'{DF}/reports/taa_v3.png', dpi=135, bbox_inches='tight')
print('\n图已存 reports/taa_v3.png')

last = R.index[-1]
print('\n当前状态（%s）：' % last)
print('  ERP %.2f%% | 分位 %.0f%% → 风险预算 %.0f%%' %
      (ERP.loc[last], ERP_Q.loc[last] * 100, budget(last) * 100))
print('  建议持仓：', {k: round(float(v), 3) for k, v in vA(last, 0).items()})
