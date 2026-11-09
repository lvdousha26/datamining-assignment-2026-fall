# -*- coding: utf-8 -*-
"""拆股/复权导致脏数据的系统审计（v2，修正停牌零价导致的 inf）

A 股有硬约束可自检 —— 涨跌停：
    主板 ±10%｜创业板 2020-08-24 前 ±10%、之后 ±20%｜科创板 ±20%｜北交所 ±30%
超出上限的日收益只可能来自：① 新股前 5 日 ② 停牌复牌跳空 ③ 复权断层/数据错误。
剔除 ①② 后剩下的就是 ③。

同时检验两个独立字段的一致性：close 反推收益 vs 东财 pctChg。
回测矩阵 R 用的是 pctChg，若 pctChg 干净而 close 脏，说明我们的模型没被污染。
"""
import glob, os
import numpy as np, pandas as pd

ROOT = 'D:/quant'
pd.set_option('display.width', 200)


def limit_of(sym, dt):
    s = str(sym).zfill(6)
    if s.startswith('688'): return 0.20
    if s.startswith('30'):  return 0.20 if dt >= pd.Timestamp('2020-08-24') else 0.10
    if s[0] in '84':        return 0.30
    return 0.10


print('读取行情…', flush=True)
px = pd.concat([pd.read_parquet(f, columns=['date', 'symbol', 'close', 'pctChg'])
                for f in sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet'))], ignore_index=True)
px['date'] = pd.to_datetime(px['date'])
px['symbol'] = px['symbol'].astype(str).str.zfill(6)
px = px.sort_values(['symbol', 'date']).reset_index(drop=True)

n0 = len(px)
nzero = int((px['close'] <= 0).sum())
print('  %d 行 | %d 只 | close<=0（停牌）%d 行 = %.2f%%'
      % (n0, px.symbol.nunique(), nzero, nzero / n0 * 100), flush=True)

# 关键修正：先把 close<=0 置 NaN，再做 pct_change，否则分母为 0 产生 inf
px['c'] = px['close'].where(px['close'] > 0)
px['r_close'] = px.groupby('symbol')['c'].transform(lambda s: s.pct_change())
px['r_pct'] = px['pctChg'] / 100.0

px['prev_date'] = px.groupby('symbol')['date'].shift(1)
px['gap_days'] = (px['date'] - px['prev_date']).dt.days
px['age'] = px.groupby('symbol').cumcount()

d = px.dropna(subset=['r_close', 'r_pct']).copy()
d = d[np.isfinite(d.r_close)]
d['diff'] = (d.r_close - d.r_pct).abs()

print('\n=== ① close 反推 vs pctChg ===')
print('  有效样本 %d | 相关 %.6f | |差|>0.5%% 占 %.4f%% | 最大差 %.1f%%'
      % (len(d), d.r_close.corr(d.r_pct), (d['diff'] > 0.005).mean() * 100, d['diff'].max() * 100))

d['lim'] = np.array([limit_of(s, t) for s, t in zip(d.symbol, d.date)])
d['exempt'] = (d['age'] < 5) | (d['gap_days'] > 10)

for tag, col in [('close 反推', 'r_close'), ('pctChg 字段', 'r_pct')]:
    bad = d[(d[col].abs() > d['lim'] + 0.02) & ~d['exempt']]
    print('\n=== ② 涨跌幅越界 —— %s ===' % tag)
    print('  %d 条，占有效样本 %.4f%%，涉及 %d 只' % (len(bad), len(bad) / len(d) * 100, bad.symbol.nunique()))
    if len(bad):
        print('  收益：均值 %+.1f%% | 中位 %+.1f%% | 区间 [%.0f%%, %+.0f%%]'
              % (bad[col].mean() * 100, bad[col].median() * 100, bad[col].min() * 100, bad[col].max() * 100))
        print('  按年：' + ' '.join('%d:%d' % (y, n) for y, n in bad.groupby(bad.date.dt.year).size().items()))
        print('  越界最多的 5 只：')
        t = bad.groupby('symbol').agg(n=(col, 'size'), mean_r=(col, 'mean'),
                                      mn=(col, 'min'), mx=(col, 'max'))
        t = t.sort_values('n', ascending=False).head(5)
        for c in ['mean_r', 'mn', 'mx']: t[c] = (t[c] * 100).round(1)
        print(t.to_string())

# 回测矩阵是否被污染
print('\n=== ③ 回测矩阵 R（来自 pctChg）===')
dr = np.load(f'{ROOT}/data/big/daily_ret.npz')
R = dr['R']
fin = R[np.isfinite(R)]
print('  非空 %.1f%% | 区间 [%.1f%%, %.1f%%] | |r|>0.25 的占比 %.4f%%'
      % (np.isfinite(R).mean() * 100, fin.min() * 100, fin.max() * 100, (np.abs(fin) > 0.25).mean() * 100))
print('  （构造时已把 |r|>=50%% 置 NaN，所以极限值被截断在上限内）')

# ---------- ④ 美股：qfq vs hfq ----------
print('\n=== ④ 美股：前复权 vs 后复权 ===')
US = {'105.AMZN': 'AMZN', '106.MSFT': 'MSFT', '106.JNJ': 'JNJ', '105.AAPL': 'AAPL',
      '106.XOM': 'XOM', '105.NVDA': 'NVDA', '106.V': 'V', '106.JPM': 'JPM'}
import akshare as ak
print('%-6s %10s %10s %10s %10s' % ('ticker', 'qfq最小价', 'qfq异常', 'hfq最小价', 'hfq异常'))
for code, tk in US.items():
    row = [tk]
    for adj in ['qfq', 'hfq']:
        try:
            dd = ak.stock_us_hist(symbol=code, period='daily', start_date='20040101',
                                  end_date='20261107', adjust=adj)
            s = dd.set_index(pd.to_datetime(dd['日期']))['收盘'].astype(float).sort_index()
            r = s.pct_change()
            row += [s.min(), int((r.abs() > 0.35).sum())]
        except Exception:
            row += [np.nan, np.nan]
    print('%-6s %10.2f %10.0f %10.2f %10.0f' % tuple(row))
