# -*- coding: utf-8 -*-
"""退市风险压力测试 + 流动性/市值过滤
① 在回测里注入退市风险：每年 h 概率退市，损失 LGD
② 加流动性下限（剔除成交额最低 X%），看代价有多大
③ 看当前策略持仓的市值/流动性画像
"""
import numpy as np, pandas as pd

ROOT = r'D:/quant'
LIB = ['INTRADAY20', 'MOM120', 'MOM60', 'QUAD_ASYM', 'REV5', 'SIZE20', 'VOL20', 'OVERNIGHT20']
DIRN = {'INTRADAY20': -1, 'MOM120': -1, 'MOM60': -1, 'QUAD_ASYM': -1,
        'REV5': 1, 'SIZE20': 1, 'VOL20': 1, 'OVERNIGHT20': 1}
COST = 2e-3

panel = pd.read_parquet(f'{ROOT}/data/panel.parquet')[['date', 'symbol', 'label', 'amount']]
df = None
for f in LIB:
    x = pd.read_parquet(f'{ROOT}/data/factors/{f}.parquet')[['date', 'symbol', 'value']]
    x['value'] = x['value'] * DIRN[f]
    x = x.rename(columns={'value': f})
    df = x if df is None else df.merge(x, on=['date', 'symbol'], how='inner')
df = df.merge(panel, on=['date', 'symbol'], how='inner').dropna()
for f in LIB:
    df['r_'+f] = df.groupby('date')[f].transform(lambda s: s.rank(pct=True))

tr = df[df['date'] <= '2023-12-31']
icir = {}
for f in LIB:
    s = tr.groupby('date').apply(lambda g: g['r_'+f].corr(g['label'], method='spearman')).dropna()
    icir[f] = s.mean()/s.std()
w = pd.Series({f: max(icir[f], 0) for f in LIB}); w /= w.sum()
df['score'] = sum(df['r_'+f]*w[f] for f in LIB)
df['liq_pct'] = df.groupby('date')['amount'].transform(lambda s: s.rank(pct=True))

def bt(top=0.10, buffer=0.30, liq_floor=0.0, hazard=0.0, lgd=0.85, cost=COST):
    d0 = df[df['liq_pct'] >= liq_floor] if liq_floor > 0 else df
    dates = np.sort(d0['date'].unique()); reb = dates[::5]
    prev = set(); rets = []; caps = []
    h_per = 1 - (1 - hazard) ** (5/252)
    for i, d in enumerate(reb[:-1]):
        g = d0[d0['date'] == d].set_index('symbol')
        s = g['score'].dropna(); n = max(int(len(s)*top), 5)
        rk = s.rank(pct=True)
        if prev and buffer > 0:
            keep = set(rk[rk >= 1-buffer].index) & prev
            need = n - len(keep)
            add = rk[(rk >= 1-top) & (~rk.index.isin(keep))].nlargest(max(need, 1)).index
            hold = list(keep | set(add))[:n]
        else:
            hold = list(rk.nlargest(n).index)
        nxt = d0[d0['date'] == reb[i+1]]
        r = nxt[nxt['symbol'].isin(hold)]['label'].mean()
        t = len(set(hold) - prev)/max(len(hold), 1) if prev else 1.0
        drag = hazard and h_per*lgd          # 期望年化损耗摊到每期
        rets.append((reb[i+1], r - t*cost - (h_per*lgd if hazard else 0)))
        caps.append(g.loc[[c for c in hold if c in g.index], 'amount'].median())
        prev = set(hold)
    s = pd.Series(dict(rets)).sort_index()
    return s, np.median(caps)

def st(r):
    a = (1+r).prod()**(50.4/len(r))-1; v = r.std()*np.sqrt(50.4); nav = (1+r).cumprod()
    return a, (a-.02)/v, (nav/nav.cummax()-1).min()

print('=== ① 持仓画像（当前全池 top10%）===')
_, cap = bt()
allmed = df.groupby('date')['amount'].median().median()
print('  持仓中位成交额 %.0f 万元 | 全池中位 %.0f 万元 → 持仓流动性是池中位数的 %.2f 倍' % (
    cap/1e4, allmed/1e4, cap/allmed))

print('\n=== ② 退市冲击压力测试（top10%+buffer30%）===')
print('  年退市率   年化    夏普   回撤')
for h in [0.0, 0.01, 0.02, 0.03, 0.05]:
    s, _ = bt(hazard=h)
    a, sh, dd = st(s)
    print('  %5.1f%%    %6.2f%%  %5.2f  %6.1f%%' % (h*100, a*100, sh, dd*100))
print('  （LGD=-85%，退市率 5%% 约等于"池里 1/20 的票每年退一家"）')

print('\n=== ③ 流动性/市值下限过滤（hazard=2%%）===')
print('  剔除最低   年化    夏普   回撤   持仓中位成交额')
for lf in [0.0, 0.2, 0.4, 0.6, 0.8]:
    s, c = bt(liq_floor=lf, hazard=0.02)
    a, sh, dd = st(s)
    print('  %5.0f%%     %6.2f%%  %5.2f  %6.1f%%   %.0f 万' % (lf*100, a*100, sh, dd*100, c/1e4))

print('\n=== ④ 基准对照（等权池，含 hazard=2%）===')
ew = df.groupby('date')['label'].mean()
ewd = ew[ew.index.isin(pd.Index(np.sort(df['date'].unique())[::5]))].dropna()
h_per = 1-(1-0.02)**(5/252)
a, sh, dd = st(ewd - h_per*0.85)
print('  等权池费后年化 %.2f%% 夏普 %.2f' % (a*100, sh))
