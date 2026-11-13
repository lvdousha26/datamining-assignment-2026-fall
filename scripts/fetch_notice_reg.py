# -*- coding: utf-8 -*-
"""抓「监管类公告」（问询/立案/处罚/警示）——只抓建模用的抽样日

数据源：东财公告列表 `stock_notice_report(symbol='全部', date=YYYYMMDD)`，
一次返回当天全部公司的公告标题，按关键词过滤后落盘。

关键词：问询、监管、立案、处罚、警示、关注函、整改、违规、退市风险
输出：data/big/notice_reg/{YYYY-MM-DD}.parquet，列 = symbol, n_reg, titles

用法：python scripts/fetch_notice_reg.py --step 5 --start 2019-01-01
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
OUT = f'{ROOT}/data/big/notice_reg'
os.makedirs(OUT, exist_ok=True)
KW = ['问询', '监管', '立案', '处罚', '警示', '关注函', '整改', '违规', '退市风险', '风险提示']


def one(d):
    f = f'{OUT}/{d}.parquet'
    if os.path.exists(f) and os.path.getsize(f) > 0:
        return 'skip'
    import akshare as ak
    for k in range(3):
        try:
            x = ak.stock_notice_report(symbol='全部', date=d.replace('-', ''))
            break
        except Exception:
            x = None
            time.sleep(2 * (k + 1))
    if x is None or len(x) == 0:
        return 'empty'
    x = x.copy()
    x['symbol'] = x['代码'].astype(str).str.zfill(6)
    hit = x['公告标题'].astype(str).apply(lambda t: any(k in t for k in KW))
    x = x[hit]
    if len(x) == 0:
        df = pd.DataFrame({'symbol': ['000000'], 'n_reg': [0], 'titles': ['']})   # 占位，表示当天已抓且无命中
    else:
        df = x.groupby('symbol').agg(n_reg=('公告标题', 'size'),
                                     titles=('公告标题', lambda s: '|'.join(s)[:300])).reset_index()
    tmp = f + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, f)
    return 'ok'


def main(step=5, start='2019-01-01', workers=6):
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    dates = pd.to_datetime(pd.Series(d['dates']).astype(str))
    ds = [str(x.date()) for x in dates if x >= pd.Timestamp(start)][::step]
    print('抓取 %d 个抽样日的监管公告（每 %d 日一次）' % (len(ds), step), flush=True)
    t0 = time.time()
    with Pool(workers) as p:
        for i, st in enumerate(p.imap_unordered(one, ds, chunksize=4), 1):
            if i % 40 == 0:
                print('  %d/%d | %.0fs' % (i, len(ds), time.time() - t0), flush=True)
    print('完成 %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    with RunLogger(__file__, tag='notice_reg'):
        main(a.step, a.start, a.workers)
