# -*- coding: utf-8 -*-
"""抓两融标的名单（逐日，可断点续传）

用途：构建「纳入/调出两融标的」事件表，做事件研究。
  深交所：`stock_margin_underlying_info_szse(date)` → 证券代码 + 当日可融资/可融券
  上交所：`stock_margin_detail_sse(date)`         → 标的证券代码（当日有融资余额的标的）

输出：data/big/margin/{YYYY-MM-DD}.parquet，列 = symbol, exchange
      （已过滤：仅当该股当日确实在标的名单/有融资余额时记录）

用法：python scripts/fetch_margin.py --start 2016-01-01 --workers 6
"""
import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
OUT = f'{ROOT}/data/big/margin'
os.makedirs(OUT, exist_ok=True)


def one(args):
    d = args
    f = f'{OUT}/{d}.parquet'
    if os.path.exists(f) and os.path.getsize(f) > 0:
        return (d, 'skip', 0)
    import akshare as ak
    rows = []
    for k in range(3):
        try:
            a = ak.stock_margin_underlying_info_szse(date=d.replace('-', ''))
            if len(a):
                s = a['证券代码'].astype(str).str.zfill(6)
                m = (a.get('当日可融资', pd.Series(1, index=a.index)).astype(str) != '否')
                rows += [(x, 'SZ') for x in s[m]]
            break
        except Exception:
            time.sleep(1.5 * (k + 1))
    for k in range(3):
        try:
            b = ak.stock_margin_detail_sse(date=d.replace('-', ''))
            if len(b):
                rows += [(str(x).zfill(6), 'SH') for x in b['标的证券代码']]
            break
        except Exception:
            time.sleep(1.5 * (k + 1))
    if not rows:
        return (d, 'empty', 0)
    df = pd.DataFrame(rows, columns=['symbol', 'exchange']).drop_duplicates()
    tmp = f + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, f)
    return (d, 'ok', len(df))


def main(start='2016-01-01', workers=6):
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    dates = pd.to_datetime(pd.Series(d['dates']).astype(str))
    todo = [str(x.date()) for x in dates if x >= pd.Timestamp(start)]
    print('待抓 %d 个交易日（%s ~ %s）| workers=%d' % (len(todo), todo[0], todo[-1], workers), flush=True)

    t0 = time.time()
    ok = skip = empty = 0
    with Pool(workers) as p:
        for i, (dd, st, n) in enumerate(p.imap_unordered(one, todo, chunksize=8), 1):
            if st == 'ok':
                ok += 1
            elif st == 'skip':
                skip += 1
            else:
                empty += 1
            if i % 100 == 0:
                print('  %d/%d | 新抓 %d 跳过 %d 空 %d | %.0fs' % (
                    i, len(todo), ok, skip, empty, time.time() - t0), flush=True)
    print('\n完成：新抓 %d / 跳过 %d / 空 %d | 用时 %.0fs' % (ok, skip, empty, time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    with RunLogger(__file__, tag='margin'):
        main(a.start, a.workers)
