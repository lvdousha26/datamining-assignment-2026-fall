# -*- coding: utf-8 -*-
"""合成最终 ST 日历：baostock 精确 isST ∪ 涨跌幅限制代理（补全并发失败缺口）

baostock 并发被服务端限流，只成功查询到部分股票；
代理用「滚动 20 日 max|pctChg|<=5.5% 且精确触及 5% 至少 2 次」识别 5% 涨跌幅限制期，
在已查询样本上 precision 0.89 / recall 0.37（见 reports/st_proxy_check.log）。
取并集 = 宁可多剔，不漏剔；来源写在 source 列便于审计。
"""
import glob
import numpy as np, pandas as pd

ROOT = r'D:/quant'
files = sorted(glob.glob(f'{ROOT}/data/big/st/shard_*.parquet'))
conf = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates()
conf['date'] = pd.to_datetime(conf['date'])
conf['source'] = 'baostock'
print('baostock 精确 ST: %d 行 | %d 只 | %d 个分片' %
      (len(conf), conf.symbol.nunique(), len(files)), flush=True)

bad = glob.glob(f'{ROOT}/data/big/st/bad_*.csv')
nbad = sum(len(pd.read_csv(f, header=None)) for f in bad) if bad else 0
print('baostock 查询失败 %d 只（由代理补全）' % nbad, flush=True)

raw = pd.concat([pd.read_parquet(f, columns=['date', 'symbol', 'pctChg'])
                 for f in sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet'))],
                ignore_index=True)
raw['date'] = pd.to_datetime(raw['date'])
raw['symbol'] = raw['symbol'].astype(str).str.zfill(6)
raw['a'] = pd.to_numeric(raw['pctChg'], errors='coerce').abs()
raw = raw.sort_values(['symbol', 'date']).reset_index(drop=True)
gs = raw.groupby('symbol', sort=False)['a']
raw['mx20'] = gs.transform(lambda x: x.rolling(20, min_periods=10).max())
raw['hit20'] = raw['a'].between(4.9, 5.1).groupby(raw['symbol']).transform(
    lambda x: x.rolling(20, min_periods=10).sum())
proxy = raw.loc[(raw['mx20'] <= 5.5) & (raw['hit20'] >= 2), ['date', 'symbol']].copy()
proxy['source'] = 'proxy'
print('涨跌幅代理 ST: %d 行 | %d 只 | 占全样本 %.2f%%' %
      (len(proxy), proxy.symbol.nunique(), len(proxy) / len(raw) * 100), flush=True)

# 主口径：只用 baostock 精确集。代理误报会剔掉低波动股，引入新的选择偏差，
# 因此仅另存宽口径供敏感性检验，不进主面板。
conf.to_parquet(f'{ROOT}/data/big/st_calendar.parquet', index=False)
allst = pd.concat([conf, proxy], ignore_index=True)
wide = allst.groupby(['date', 'symbol'])['source'].agg(
    lambda s: 'both' if set(s) == {'baostock', 'proxy'} else s.iloc[0]).reset_index()
wide.to_parquet(f'{ROOT}/data/big/st_calendar_wide.parquet', index=False)
print('\n主口径(baostock 精确) %d 行 | %d 只' % (len(conf), conf.symbol.nunique()), flush=True)
print('宽口径(含代理, 仅敏感性检验) %d 行 | %d 只 | 来源 %s' %
      (len(wide), wide.symbol.nunique(), wide.source.value_counts().to_dict()), flush=True)
print('按年 ST 只数:', conf.groupby(conf.date.dt.year)['symbol'].nunique().to_dict(), flush=True)
print('已存 st_calendar.parquet / st_calendar_wide.parquet')
