# -*- coding: utf-8 -*-
"""ST 剔除的敏感性检验（用 v1 备份面板，不干扰正在进行的 v2 重训）
回答：v2 因 baostock 并发限流只拿到部分历史 ST，这个覆盖缺口会不会改变结论？
对比三种口径的测试段 IC：不剔 / 剔精确集 / 剔宽口径(含代理)。
"""
import numpy as np, pandas as pd
from scipy import stats as sps

ROOT = r'D:/quant'
z = np.load(f'{ROOT}/data/big/v1/arr.npz', allow_pickle=True)
Y2 = z['Y2']
dates = pd.to_datetime(z['dates']); syms = np.array([str(s) for s in z['symbols']])
di = {d: i for i, d in enumerate(dates)}; si = {s: i for i, s in enumerate(syms)}
p = np.load(f'{ROOT}/data/big/v1/preds.npz')
tt, ii, pred = p['tt_te'], p['ii_te'], p['p_st_te']

def keyset(fn):
    d = pd.read_parquet(f'{ROOT}/data/big/{fn}')
    d['date'] = pd.to_datetime(d['date'])
    out = set()
    for dt, s in zip(d['date'], d['symbol'].astype(str).str.zfill(6)):
        t, i = di.get(dt), si.get(s)
        if t is not None and i is not None:
            out.add((t, i))
    return out

strict = keyset('st_calendar.parquet')
wide = keyset('st_calendar_wide.parquet')
print('精确 ST 命中测试段 (t,i) %d 个 | 宽口径 %d 个 | 测试样本 %d 行' %
      (len(strict & set(zip(tt, ii))), len(wide & set(zip(tt, ii))), len(tt)), flush=True)

def ic(mask):
    d = pd.DataFrame({'d': tt[mask], 'p': pred[mask], 'y': Y2[tt[mask], ii[mask]]})
    s = d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    return s.mean(), s.mean() / s.std(), mask.mean()

allk = set(zip(tt, ii))
for tag, ks in [('不剔 ST(v1 原口径)', set()),
                ('剔 baostock 精确集(v2 主口径)', strict),
                ('剔宽口径(含代理)', wide)]:
    mask = np.array([(t, i) not in ks for t, i in zip(tt, ii)], dtype=bool)
    a, b, frac = ic(mask)
    print('  %-28s IC %+.4f  ICIR %+.2f  保留样本 %.2f%%' % (tag, a, b, frac * 100))
