# -*- coding: utf-8 -*-
"""补抓「融资余额」——只抓建模要用的那些日子（每 5 个交易日一次）

`fetch_margin.py` 只存了 symbol/exchange，做「两融余额占成交额比」需要余额本身。
这里按需补抓，避免重跑全量。

输出：data/big/margin_bal/{YYYY-MM-DD}.parquet，列 = symbol, bal(融资余额, 元)
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
OUT = f'{ROOT}/data/big/margin_bal'
os.makedirs(OUT, exist_ok=True)


def one(d):
    f = f'{OUT}/{d}.parquet'
    if os.path.exists(f) and os.path.getsize(f) > 0:
        return 'skip'
    import akshare as ak
    rows = []
    for fn, code_col, bal_col in [('stock_margin_detail_szse', '证券代码', '融资余额'),
                                  ('stock_margin_detail_sse', '标的证券代码', '融资余额')]:
        for k in range(3):
            try:
                x = getattr(ak, fn)(date=d.replace('-', ''))
                if len(x):
                    rows.append(pd.DataFrame({
                        'symbol': x[code_col].astype(str).str.zfill(6),
                        'bal': pd.to_numeric(x[bal_col], errors='coerce')}))
                break
            except Exception:
                time.sleep(1.2 * (k + 1))
    if not rows:
        return 'empty'
    df = pd.concat(rows).dropna().groupby('symbol', as_index=False)['bal'].max()
    tmp = f + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, f)
    return 'ok'


def main(step=5, start='2019-01-01', workers=8):
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    dates = pd.to_datetime(pd.Series(d['dates']).astype(str))
    ds = [str(x.date()) for x in dates if x >= pd.Timestamp(start)][::step]
    print('补抓 %d 个日期（每 %d 日取一次）' % (len(ds), step), flush=True)
    t0 = time.time()
    with Pool(workers) as p:
        for i, st in enumerate(p.imap_unordered(one, ds, chunksize=8), 1):
            if i % 60 == 0:
                print('  %d/%d | %.0fs' % (i, len(ds), time.time() - t0), flush=True)
    print('完成 %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--start', default='2019-01-01')
    a = ap.parse_args()
    with RunLogger(__file__, tag='margin_bal'):
        main(a.step, a.start)
