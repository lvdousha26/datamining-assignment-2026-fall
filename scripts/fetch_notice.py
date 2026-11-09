# -*- coding: utf-8 -*-
"""抓 A 股公告，只保留「减持 / 增持」相关（供事件研究用）

按年分片，每年一个 parquet，天然支持断点续传（已存在的年份跳过）。
只存标题命中关键词的记录，落盘很小（全市场全量公告约 400 万条，筛后约 8 万条）。

用法
    python scripts/fetch_notice.py              # 抓全部年份
    python scripts/fetch_notice.py --years 2020,2021
    python scripts/fetch_notice.py --workers 4
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = r'D:/quant'
OUT = f'{ROOT}/data/big/notice'
KEY = '减持|增持'


def trading_days():
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    return pd.to_datetime(pd.Series(d['dates']).astype(str))


def fetch_year(year, days):
    """抓某一年，返回筛选后的 DataFrame"""
    import akshare as ak
    rows, fails = [], 0
    for d in days:
        ds = d.strftime('%Y%m%d')
        for attempt in range(2):
            try:
                x = ak.stock_notice_report(symbol='全部', date=ds)
                if len(x):
                    m = x[x['公告标题'].astype(str).str.contains(KEY, na=False)]
                    if len(m):
                        m = m.copy()
                        m['date'] = d.strftime('%Y-%m-%d')
                        rows.append(m[['date', '代码', '名称', '公告标题', '公告类型']])
                break
            except Exception:
                if attempt == 1:
                    fails += 1
                time.sleep(0.5)
        time.sleep(0.12)
    print(f'  {year}: {len(days)} 个交易日 | 命中 {sum(len(r) for r in rows)} 条 | 失败 {fails} 天',
          flush=True)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default=None)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    td = trading_days()
    years = [int(x) for x in a.years.split(',')] if a.years else sorted(td.dt.year.unique())
    todo = []
    for y in years:
        f = f'{OUT}/notice_{y}.parquet'
        if os.path.exists(f):
            print(f'  {y}: 已存在，跳过', flush=True)
            continue
        todo.append((y, td[td.dt.year == y].tolist()))

    print(f'待抓 {len(todo)} 年 | {a.workers} 进程', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fut = {ex.submit(fetch_year, y, ds): y for y, ds in todo}
        for f in as_completed(fut):
            y = fut[f]
            try:
                df = f.result()
                if df is not None and len(df):
                    df.to_parquet(f'{OUT}/notice_{y}.parquet', index=False)
                    print(f'  {y}: 已存 {len(df)} 条', flush=True)
            except Exception as e:
                print(f'  {y}: 失败 {str(e)[:60]}', flush=True)

    fs = sorted(glob.glob(f'{OUT}/notice_*.parquet'))
    if fs:
        all_df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        all_df.to_parquet(f'{OUT}/notice_all.parquet', index=False)
        print(f'\n合计 {len(all_df):,} 条 | {all_df.date.nunique()} 个交易日 | '
              f'{all_df["代码"].nunique()} 只股票 | 耗时 {time.time() - t0:.0f}s')
        print(all_df['公告类型'].value_counts().head(8).to_string())
