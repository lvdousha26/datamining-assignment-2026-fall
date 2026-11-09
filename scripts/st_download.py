# -*- coding: utf-8 -*-
"""时变 ST 日历下载（baostock isST 字段，真正的历史 ST 状态）
用法：python scripts/st_download.py <shard_id> <n_shards>
每个进程独立 login，只落盘 isST==1 的 (date, symbol)，最后用 st_merge.py 合并。
"""
import os, sys, glob, time
import numpy as np, pandas as pd
import baostock as bs

ROOT = r'D:/quant'
OUT = f'{ROOT}/data/big/st'
os.makedirs(OUT, exist_ok=True)
sid, nsh = int(sys.argv[1]), int(sys.argv[2])
START, END = '2015-01-01', '2026-10-31'

files = sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet'))
codes = sorted(set().union(*[
    set(pd.read_parquet(f, columns=['symbol'])['symbol'].astype(str).str.zfill(6))
    for f in files]))
mine = codes[sid::nsh]

def bcode(c):
    return ('sh.' if c[0] == '6' else 'sz.') + c

lg = bs.login()
print('shard %d/%d login %s | %d 只' % (sid, nsh, lg.error_code, len(mine)), flush=True)
rows, bad = [], []
t0 = time.time()
for j, c in enumerate(mine):
    try:
        rs = bs.query_history_k_data_plus(
            bcode(c), 'date,isST', start_date=START, end_date=END,
            frequency='d', adjustflag='2')
        if rs.error_code != '0':
            bad.append(c); continue
        while rs.next():
            r = rs.get_row_data()
            if r[1] == '1':
                rows.append((r[0], c))
    except Exception as e:
        bad.append(c)
    if (j + 1) % 100 == 0:
        print('  shard%d %d/%d  ST行 %d  %.0fs' %
              (sid, j + 1, len(mine), len(rows), time.time() - t0), flush=True)
bs.logout()

d = pd.DataFrame(rows, columns=['date', 'symbol'])
d['date'] = pd.to_datetime(d['date'])
d = d.drop_duplicates()
d.to_parquet(f'{OUT}/shard_{sid:02d}.parquet', index=False)
pd.Series(bad).to_csv(f'{OUT}/bad_{sid:02d}.csv', index=False, header=False)
print('shard%d 完成：ST 行 %d | 涉及 %d 只 | 失败 %d | %.0fs' %
      (sid, len(d), d.symbol.nunique(), len(bad), time.time() - t0), flush=True)
