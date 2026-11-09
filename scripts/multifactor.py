# -*- coding: utf-8 -*-
"""多因子策略：5 种合成方法对比 + 组合构建 + 成本/换手控制
纪律：因子权重只用训练期(<2024)的 ICIR 估计；OOS(2024+) 只做评估
组合：周频(5交易日)调仓，top20% 等权，rank buffer（留在前40%就不换）
"""
import io, contextlib
import numpy as np, pandas as pd
from scipy import stats as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'D:/quant'
LIB = ['INTRADAY20', 'MOM120', 'MOM60', 'QUAD_ASYM', 'REV5', 'SIZE20', 'VOL20', 'OVERNIGHT20']
DIRN = {'INTRADAY20': -1, 'MOM120': -1, 'MOM60': -1, 'QUAD_ASYM': -1,
        'REV5': 1, 'SIZE20': 1, 'VOL20': 1, 'OVERNIGHT20': 1}
COST = 2e-3

panel = pd.read_parquet(f'{ROOT}/data/panel.parquet')[['date', 'symbol', 'label']]
df = None
for f in LIB:
    x = pd.read_parquet(f'{ROOT}/data/factors/{f}.parquet')[['date', 'symbol', 'value']]
    x['value'] = x['value'] * DIRN[f]
    x = x.rename(columns={'value': f})
    df = x if df is None else df.merge(x, on=['date', 'symbol'], how='inner')
df = df.merge(panel, on=['date', 'symbol'], how='inner').dropna()
for f in LIB:                                   # 逐日截面 rank（0-1，越大越好）
    df['r_'+f] = df.groupby('date')[f].transform(lambda s: s.rank(pct=True))
print('样本 %d 行 | %d 日 | %d 只' % (len(df), df['date'].nunique(), df['symbol'].nunique()))

TR_END, OOS_START = '2023-12-31', '2024-01-01'
tr = df[df['date'] <= TR_END]

# 训练期 ICIR（用于权重，绝不用 OOS）
def daily_ic_series(d, col, ycol='label'):
    s = d.groupby('date').apply(lambda g: sps.spearmanr(g[col], g[ycol]).correlation).dropna()
    return s
icir = {}
for f in LIB:
    s = daily_ic_series(tr, 'r_'+f)
    icir[f] = s.mean()/s.std()
print('\n训练期 ICIR（含符号，全部应为正）:')
for k, v in sorted(icir.items(), key=lambda x: -x[1]):
    print('  %-12s %+6.3f' % (k, v))

# ---- 5 种合成 ----
w_eq = pd.Series({f: 1/len(LIB) for f in LIB})
w_icir = pd.Series({f: max(icir[f], 0) for f in LIB}); w_icir /= w_icir.sum()
top4 = sorted(LIB, key=lambda f: -icir[f])[:4]
w_top4 = pd.Series({f: (1/4 if f in top4 else 0) for f in LIB})
w_nosz = w_icir.copy(); w_nosz['SIZE20'] = 0; w_nosz /= w_nosz.sum()
print('\nICIR 权重:', {k: round(v, 3) for k, v in w_icir.items()})
print('Top4 因子:', top4)
print('去 SIZE20 权重:', {k: round(v, 3) for k, v in w_nosz.items() if v > 0})

for nm, w in [('EQ', w_eq), ('ICIR', w_icir), ('TOP4', w_top4), ('noSIZE', w_nosz)]:
    df[nm] = sum(df['r_'+f]*w[f] for f in LIB)
# 中性化版本：对 SIZE20 做截面回归取残差（消除市值暴露）
def resid_score(d):
    x = d['SIZE20'].values; y = d['ICIR'].values
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return pd.Series(y - A @ b, index=d.index)
df['NEUT'] = df.groupby('date', group_keys=False).apply(resid_score)

# ---- 回测 ----
def backtest(score_col, top=0.2, buffer=0.4, cost=COST):
    dates = np.sort(df['date'].unique())
    reb = dates[::5]                       # 每5个交易日调仓
    prev = set(); rets, turns, nhold = [], [], []
    for i, d in enumerate(reb[:-1]):
        g = df[df['date'] == d].set_index('symbol')
        if len(g) < 50: continue
        s = g[score_col].dropna()
        n = max(int(len(s)*top), 5)
        rk = s.rank(pct=True)
        if prev:
            keep = set(rk[rk >= 1-buffer].index) & prev        # buffer：还在前40%(高分)就留
            need = n - len(keep)
            add = rk[(rk >= 1-top) & (~rk.index.isin(keep))].nlargest(max(need,1)).index
            hold = list(keep | set(add))
        else:
            hold = list(rk.nlargest(n).index)
        hold = hold[:n]
        nxt = df[df['date'] == reb[i+1]]
        r = nxt[nxt['symbol'].isin(hold)]['label'].mean()
        t = len(set(hold) - prev)/max(len(hold), 1) if prev else 1.0
        rets.append((reb[i+1], r - t*cost*2))
        turns.append(t); nhold.append(len(hold)); prev = set(hold)
    s = pd.Series(dict(rets)).sort_index()
    return s, np.mean(turns), np.mean(nhold)

def stats(r, tag):
    r = r.dropna(); ann = (1+r).prod()**(50.4/len(r))-1
    vol = r.std()*np.sqrt(50.4); nav = (1+r).cumprod()
    dd = (nav/nav.cummax()-1).min()
    print('  %-10s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%% 胜率 %2.0f%%' % (
        tag, ann*100, vol*100, (ann-.02)/vol, dd*100, (r>0).mean()*100))
    return ann, (ann-.02)/vol, dd, nav

print('\n=== 全样本（2019-2026）===')
res = {}
for col in ['EQ', 'ICIR', 'TOP4', 'noSIZE', 'NEUT']:
    r, to, nh = backtest(col)
    a, sh, dd, nav = stats(r, col)
    res[col] = (r, nav, a)
    print('             换手/期 %.0f%% 持仓 %.0f 只' % (to*100, nh))
# 基准：等权全市场
ew = df.groupby('date')['label'].mean()
ew = ew.loc[ew.index.isin(pd.Index(sorted(df['date'].unique())[::5]))]
stats(ew.dropna(), '等权基准')

print('\n=== OOS 2024+ ===')
for col in ['EQ', 'ICIR', 'TOP4', 'noSIZE', 'NEUT']:
    r = res[col][0]
    stats(r[r.index >= OOS_START], col)
stats(ew[ew.index >= OOS_START].dropna(), '等权基准')

# 月度视角（OOS）
r_oos = res['ICIR'][0]; r_oos = r_oos[r_oos.index >= OOS_START]
mo = (1+r_oos).groupby(r_oos.index.to_period('M')).prod()-1
print('\nOOS 月度胜率 %.0f%%（%d 个月），最差月 %.1f%%' % ((mo>0).mean()*100, len(mo), mo.min()*100))

# ---- 图 ----
fig, ax = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[3, 1.4])
for col, c, lw in [('EQ', '#95a5a6', 1.2), ('ICIR', '#c0392b', 2.0), ('noSIZE', '#2980b9', 1.4)]:
    nav = res[col][1]
    ax[0].plot(nav.index, nav.values, lw=lw, color=c, label=col)
ewn = (1+ew.dropna()).cumprod()
ax[0].plot(ewn.index, ewn.values, lw=1.0, color='#7f8c8d', ls='--', label='等权基准')
ax[0].set_title('多因子策略净值（周频调仓，top20%%，费后20bp）', fontsize=11)
ax[0].legend(frameon=False); ax[0].grid(alpha=.25)
nav = res['ICIR'][1]
ax[1].fill_between(nav.index, (nav/nav.cummax()-1)*100, 0, color='#c0392b', alpha=.4, lw=0)
ax[1].set_ylabel('回撤 %'); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig(f'{ROOT}/reports/multifactor_nav.png', dpi=135, bbox_inches='tight')
print('\n图已保存 reports/multifactor_nav.png')

pd.DataFrame({'date': res['ICIR'][0].index, 'ret_ICIR': res['ICIR'][0].values}).to_csv(
    f'{ROOT}/reports/multifactor_returns.csv', index=False)
