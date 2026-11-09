# -*- coding: utf-8 -*-
"""用后复权（hfq）重新下载 A 股全市场行情

背景：东财前复权（qfq）是「减法复权」——用历史价格减去累计分红，
      对高分红老股会把早年价格压到 0 甚至负数（茅台 qfq 最小价 −143.50），
      导致收益率方向反转、出现 4 万处「涨跌停越界」。后复权（hfq）无此问题。

输出：data/big/raw_hfq/{symbol}.parquet，字段与旧 raw/ 一致。
断点续传：已存在的文件跳过。
"""
import os, sys, time, traceback
import numpy as np, pandas as pd
from multiprocessing import Pool
import akshare as ak

ROOT = 'D:/quant'
OUT = f'{ROOT}/data/big/raw_hfq'
os.makedirs(OUT, exist_ok=True)
COLS = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
        '成交量': 'volume', '成交额': 'amount', '振幅': 'amplitude',
        '涨跌幅': 'pctChg', '换手率': 'turn'}


def one(sym):
    f = f'{OUT}/{sym}.parquet'
    if os.path.exists(f):
        return ('skip', sym, 0)
    for k in range(3):
        try:
            d = ak.stock_zh_a_hist(symbol=sym, period='daily',
                                   start_date='20150101', end_date='20261231', adjust='hfq')
            if len(d) == 0:
                return ('empty', sym, 0)
            d = d.rename(columns=COLS)
            d['symbol'] = sym
            keep = ['date', 'symbol', 'open', 'high', 'low', 'close',
                    'volume', 'amount', 'amplitude', 'pctChg', 'turn']
            d = d[[c for c in keep if c in d.columns]]
            d['date'] = pd.to_datetime(d['date'])
            d = d.sort_values('date').reset_index(drop=True)
            if d['close'].min() <= 0:
                return ('BADNEG', sym, len(d))
            d.to_parquet(f, index=False)
            return ('ok', sym, len(d))
        except Exception as e:
            if k == 2:
                return ('fail:' + str(e)[:40], sym, 0)
            time.sleep(1.5 * (k + 1))
    return ('fail', sym, 0)


if __name__ == '__main__':
    import glob
    syms = sorted({os.path.basename(p).split('_')[1].split('.')[0].replace('.parquet', '')
                   for p in glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet')})
    # 从旧分片里取真实股票池
    old = pd.concat([pd.read_parquet(f, columns=['symbol'])
                     for f in sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet'))])
    syms = sorted(old['symbol'].astype(str).str.zfill(6).unique())
    print('股票池 %d 只' % len(syms), flush=True)
    todo = [s for s in syms if not os.path.exists(f'{OUT}/{s}.parquet')]
    print('待下载 %d 只（已完成 %d）' % (len(todo), len(syms) - len(todo)), flush=True)

    t0 = time.time()
    stat = {}
    with Pool(4) as pool:
        for i, (st, sym, n) in enumerate(pool.imap_unordered(one, todo, chunksize=8), 1):
            key = st.split(':')[0]
            stat[key] = stat.get(key, 0) + 1
            if i % 200 == 0:
                el = time.time() - t0
                print('  %d/%d | 用时 %.0fs | 预计剩余 %.0f 分钟 | %s'
                      % (i, len(todo), el, (len(todo) - i) * el / i / 60, stat), flush=True)
    print('\n完成 | %s | 总耗时 %.1f 分钟' % (stat, (time.time() - t0) / 60), flush=True)
