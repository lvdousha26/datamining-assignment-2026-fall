# -*- coding: utf-8 -*-
"""P1 因子生成（公式化 alpha，仅用 T 日及之前信息）
14 个候选因子：反转/动量/低波/流动性/隔夜-日内分解/收盘位置/量价偏离
输出：D:/quant/data/factors/{name}.parquet (date, symbol, value)
"""
import json, os, hashlib
import numpy as np, pandas as pd

ROOT = r'D:/quant'
P = pd.read_parquet(f'{ROOT}/data/panel.parquet')
P['symbol'] = P['symbol'].str.replace(r'^raw\\', '', regex=True).str.replace('.csv', '', regex=False)
P = P.sort_values(['symbol', 'date'])
print('面板 %d 行 × %d 只 | %s ~ %s' % (len(P), P['symbol'].nunique(),
      P['date'].min().date(), P['date'].max().date()))

# 转宽表便于向量化
def wide(col): return P.pivot(index='date', columns='symbol', values=col)
O, H, L, C, PC, A = wide('open'), wide('high'), wide('low'), wide('close'), wide('preclose'), wide('amount')
ret = C.pct_change()
gap = O / PC - 1                 # 隔夜
intra = C / O - 1                # 日内
vr = A / A.rolling(20).mean()    # 量比
vwap = (H + L + C) / 3

F = {}
F['REV5'] = -(C / C.shift(5) - 1)
F['REV20'] = -(C / C.shift(20) - 1)
F['MOM60'] = C / C.shift(60) - 1
F['MOM120'] = C / C.shift(120) - 1
F['MOM120_20'] = C.shift(20) / C.shift(120) - 1
F['VOL20'] = -ret.rolling(20).std()
F['VOL60'] = -ret.rolling(60).std()
F['AMIHUD20'] = (ret.abs() / A.replace(0, np.nan)).rolling(20).mean() * 1e9
F['OVERNIGHT20'] = gap.rolling(20).sum()
F['INTRADAY20'] = intra.rolling(20).sum()
F['QUAD_ASYM'] = ((gap < 0) & (intra > 0)).astype(float).rolling(20).mean() \
               - ((gap > 0) & (intra < 0)).astype(float).rolling(20).mean()
F['CLV20'] = ((C - L) / (H - L + 1e-12)).rolling(20).mean()
F['VWAPDEV20'] = ((C - vwap) / vwap).rolling(20).mean()
F['SIZE20'] = -np.log(A.rolling(20).mean() + 1)
F['HLRANGE20'] = ((H - L) / C).rolling(20).mean()

os.makedirs(f'{ROOT}/data/factors', exist_ok=True)
trials = []
for name, mat in F.items():
    long = mat.stack().rename('value').reset_index()
    long.columns = ['date', 'symbol', 'value']
    long = long[np.isfinite(long['value'])]
    long.to_parquet(f'{ROOT}/data/factors/{name}.parquet', index=False)
    trials.append(dict(factor=name, stage='P1_generated', n_trials_total=len(F),
                       panel=f'{P["date"].nunique()}d x {P["symbol"].nunique()}s'))
    print('  %-12s 覆盖 %.1f%%  均值 %.4f' % (name, 100*len(long)/len(P), long['value'].mean()))

led = f'{ROOT}/data/trials.json'
old = json.load(open(led)) if os.path.exists(led) else []
json.dump(old + trials, open(led, 'w'), ensure_ascii=False, indent=1)
print('\n已生成 %d 个因子，台账 %s 累计 %d 条' % (len(F), led, len(old)+len(trials)))
