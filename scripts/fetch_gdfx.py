# -*- coding: utf-8 -*-
"""抓「国家队」持仓：十大流通股东里的汇金 / 证金 / 社保 / 养老金

数据源：东财 `stock_gdfx_free_holding_analyse_em(date=报告期)`，一次返回全市场十大流通股东明细
        （53,604 行/期），按股东名称过滤国家队。

国家队口径（名称包含）：
  中央汇金、中国证券金融、证金、全国社保基金、基本养老保险、中国投资有限责任公司

输出：data/big/gdfx/{YYYYMMDD}.parquet，列 = symbol, holder, shares, chg, pct

用法：python scripts/fetch_gdfx.py --start 2024-03-31 --workers 4
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
OUT = f'{ROOT}/data/big/gdfx'
os.makedirs(OUT, exist_ok=True)
KW = ['中央汇金', '中国证券金融', '证金', '全国社保基金', '基本养老保险', '中国投资有限责任公司']


def one(d):
    f = f'{OUT}/{d}.parquet'
    if os.path.exists(f) and os.path.getsize(f) > 0:
        return 'skip'
    import akshare as ak
    for k in range(3):
        try:
            x = ak.stock_gdfx_free_holding_analyse_em(date=d)
            break
        except Exception:
            x = None
            time.sleep(3 * (k + 1))
    if x is None or len(x) == 0:
        return 'empty'
    x = x.copy()
    x['symbol'] = x['股票代码'].astype(str).str.zfill(6)
    hit = x['股东名称'].astype(str).apply(lambda s: any(k in s for k in KW))
    x = x[hit]
    if len(x) == 0:
        return 'nomatch'
    out = pd.DataFrame({
        'symbol': x['symbol'],
        'holder': x['股东名称'].astype(str).str[:40],
        'shares': pd.to_numeric(x['期末持股-数量'], errors='coerce'),
        'chg': pd.to_numeric(x['期末持股-数量变化'], errors='coerce'),
    })
    tmp = f + '.tmp'
    out.to_parquet(tmp, index=False)
    os.replace(tmp, f)
    return 'ok:%d' % len(out)


def main(start='2024-03-31', workers=4):
    qs = pd.date_range(start, '2025-12-31', freq='QE')
    ds = [x.strftime('%Y%m%d') for x in qs if x.strftime('%Y%m%d') <= '20250930']
    print('待抓 %d 期：%s' % (len(ds), ','.join(ds)), flush=True)
    t0 = time.time()
    with Pool(workers) as p:
        for d, st in zip(ds, p.map(one, ds)):
            print('  %s → %s (%.0fs)' % (d, st, time.time() - t0), flush=True)
    print('完成 %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2024-03-31')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    with RunLogger(__file__, tag='gdfx'):
        main(a.start, a.workers)
