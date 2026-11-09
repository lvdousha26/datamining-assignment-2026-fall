# -*- coding: utf-8 -*-
"""合并 baostock isST 分片 → 时变 ST 日历，并输出审计统计"""
import glob
import pandas as pd

ROOT = r'D:/quant'
files = sorted(glob.glob(f'{ROOT}/data/big/st/shard_*.parquet'))
print('分片 %d 个' % len(files))
d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates()
d['date'] = pd.to_datetime(d['date'])
d = d.sort_values(['date', 'symbol']).reset_index(drop=True)
d.to_parquet(f'{ROOT}/data/big/st_calendar.parquet', index=False)

bad = pd.concat([pd.read_csv(f, header=None) for f in
                 glob.glob(f'{ROOT}/data/big/st/bad_*.csv')], ignore_index=True) \
    if glob.glob(f'{ROOT}/data/big/st/bad_*.csv') else pd.DataFrame()
print('ST 日历 %d 行 | %d 只 | %s ~ %s' %
      (len(d), d.symbol.nunique(), d.date.min().date(), d.date.max().date()))
print('按年 ST 股票数:', d.groupby(d.date.dt.year)['symbol'].nunique().to_dict())
print('下载失败 %d 只（默认视为非 ST）' % len(bad))
print('已存 data/big/st_calendar.parquet')
