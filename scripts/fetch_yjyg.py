# -*- coding: utf-8 -*-
"""抓全市场业绩预告（按报告期），用于事件研究

为什么用业绩预告而不是减持公告：
  ① 预告类型是**明确的正负标签**（预增/扭亏 vs 预减/首亏），不需要 NLP
  ② 公告日期在前，正式财报在后，是「事件驱动」最标准的样本
  ③ 一个报告期一次请求，44 期搞定；减持公告要逐日拉 2600 次，且已被限流

用法
    python scripts/fetch_yjyg.py
    python scripts/fetch_yjyg.py --start 2020-01-01
"""
import argparse
import glob
import os
import time

import pandas as pd

# ---- 运行日志：统一写 logs/，不改变原有输出 ----
if __name__ == '__main__':
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from runlog import RunLogger as _RL
        _RL(__file__).__enter__()
    except Exception:
        pass

ROOT = r'D:/quant'
OUT = f'{ROOT}/data/big/yjyg'

POS = ['预增', '略增', '扭亏', '续盈']
NEG = ['预减', '首亏', '略减', '续亏']


def periods(start='2016-01-01', end='2026-12-31'):
    out = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        for md in ['0331', '0630', '0930', '1231']:
            d = pd.Timestamp(f'{y}-{md[:2]}-{md[2:]}')
            if pd.Timestamp(start) <= d <= pd.Timestamp(end):
                out.append(d.strftime('%Y%m%d'))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--end', default='2026-12-31')
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    import akshare as ak

    for p in periods(a.start, a.end):
        f = f'{OUT}/yjyg_{p}.parquet'
        if os.path.exists(f):
            continue
        for attempt in range(3):
            try:
                d = ak.stock_yjyg_em(date=p)
                if len(d):
                    d.to_parquet(f, index=False)
                    print(f'  {p}: {len(d)} 条', flush=True)
                else:
                    print(f'  {p}: 空', flush=True)
                break
            except Exception as e:
                if attempt == 2:
                    print(f'  {p}: FAIL {str(e)[:50]}', flush=True)
                time.sleep(1.5)
        time.sleep(0.3)

    fs = sorted(glob.glob(f'{OUT}/yjyg_*.parquet'))
    if not fs:
        print('无数据'); raise SystemExit
    df = pd.concat([pd.read_parquet(f).assign(period=f.split('_')[-1].replace('.parquet', ''))
                    for f in fs], ignore_index=True)
    df = df.drop_duplicates(['股票代码', '公告日期', '预告类型'])
    df.to_parquet(f'{OUT}/yjyg_all.parquet', index=False)
    print(f'\n合计 {len(df):,} 条 | {df["股票代码"].nunique()} 只 | '
          f'公告日期 {df["公告日期"].min()} ~ {df["公告日期"].max()}')
    print(df['预告类型'].value_counts().to_string())
