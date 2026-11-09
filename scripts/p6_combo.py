# -*- coding: utf-8 -*-
"""P6-lite 组合终判：8 个入池因子等权 rank 合成 → K=5 分组 + 多头超额 + 成本
口径：日频截面，T 日信号 T+1 买入持 H=5 日（用 panel label），20bp 单边
"""
import os
import numpy as np, pandas as pd

ROOT = r'D:/quant'
LIB = ['INTRADAY20', 'MOM120', 'MOM60', 'QUAD_ASYM', 'REV5', 'SIZE20', 'VOL20', 'OVERNIGHT20']
# 各因子方向（来自 OOS IC 符号）
DIRN = {'INTRADAY20': -1, 'MOM120': -1, 'MOM60': -1, 'QUAD_ASYM': -1,
        'REV5': +1, 'SIZE20': +1, 'VOL20': +1, 'OVERNIGHT20': +1}

panel = pd.read_parquet(f'{ROOT}/data/panel.parquet')[['date', 'symbol', 'label']]
comp = None
for f in LIB:
    x = pd.read_parquet(f'{ROOT}/data/factors/{f}.parquet')
    x['r'] = x.groupby('date')['value'].rank(pct=True)
    x['r'] = x['r'] * DIRN[f]
    x = x[['date', 'symbol', 'r']].rename(columns={'r': f})
    comp = x if comp is None else comp.merge(x, on=['date', 'symbol'], how='inner')

comp['score'] = comp[LIB].mean(axis=1)
m = comp.merge(panel, on=['date', 'symbol'], how='inner').dropna(subset=['score', 'label'])
print('合成因子: %d 行, %d 交易日, %d 只' % (len(m), m['date'].nunique(), m['symbol'].nunique()))

m['q'] = m.groupby('date')['score'].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
g = m.groupby('q')['label'].mean() * 100
print('\n=== K=5 分组（H=5 平均收益 %%）===')
print(g.round(2).to_string())
print('  多空(Q5-Q1) 单期 %.2f%%  → 年化约 %.1f%%（20个重叠期/年）' % (
    (g.iloc[-1] - g.iloc[0]) * 100 / 100, (g.iloc[-1]-g.iloc[0])*100*(252/5)))

# 多头组合（Q5 等权）vs 等权全市场
m['ret_q5'] = np.where(m['q'] == 4, m['label'], np.nan)
daily = m.groupby('date').agg(q5=('ret_q5', 'mean'), ew=('label', 'mean')).dropna()
tno = m[m['q'] == 4].groupby('date')['symbol'].apply(set)
turn = []
prev = None
for d, s in tno.items():
    if prev is not None: turn.append(len(s - prev) / max(len(s), 1))
    prev = s
turn = pd.Series(turn)
cost = turn.mean() * 2e-3 * (252 / 5)     # 每 5 日调仓一次
print('\n=== 多头组合（Q5 等权，持有5日）===')
for nm in ['q5', 'ew']:
    s = daily[nm]
    ann = (1 + s).prod() ** (252 / len(s)) - 1
    vol = s.std() * np.sqrt(252)
    print('  %-4s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f' % (
        nm, ann * 100, vol * 100, (ann - .02) / vol))
print('  换手: 每次调仓 %.0f%%, 年化成本约 %.2f%%' % (turn.mean() * 100, cost * 100))
net = (1 + daily['q5']).prod() ** (252 / len(daily)) - 1 - cost
print('  Q5 费后年化 %.2f%%' % (net * 100))

# OOS 分段
oos = daily[daily.index >= '2024-01-01']
if len(oos) > 100:
    print('\n=== OOS 2024+ ===')
    for nm in ['q5', 'ew']:
        s = oos[nm]; ann = (1 + s).prod() ** (252 / len(s)) - 1
        print('  %-4s 年化 %6.2f%% 夏普 %5.2f' % (nm, ann * 100, (ann - .02) / (s.std() * np.sqrt(252))))

comp.to_parquet(f'{ROOT}/data/factors/COMPOSITE8.parquet', index=False)
print('\n合成因子已存: data/factors/COMPOSITE8.parquet')
