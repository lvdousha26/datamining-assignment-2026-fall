# -*- coding: utf-8 -*-
"""Point-in-time 行业分组：用过去 250 日收益相关性做聚类，完全无前视

背景：申万官网历史分类文件不可用（swsresearch.com 连不通），
`index_component_sw` 只有当前成分，回溯到 2016 是确认的前视偏差。
替代方案：每年年初只用「上一年及以前」的日收益，PCA 降维后 KMeans 分 31 组，
标签用于接下来一整年。分组由历史数据决定，不含任何未来信息。
"""
import glob, time
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

ROOT = r'D:/quant'
OUT = f'{ROOT}/data/big/ind_roll.parquet'
K, WIN, MINOBS, NPC = 31, 250, 120, 10

t0 = time.time()
raw = pd.concat([pd.read_parquet(f, columns=['date', 'symbol', 'close'])
                 for f in sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet'))],
                ignore_index=True)
raw['date'] = pd.to_datetime(raw['date'])
raw['symbol'] = raw['symbol'].astype(str).str.zfill(6)
raw['close'] = pd.to_numeric(raw['close'], errors='coerce')
raw.loc[raw['close'] <= 0, 'close'] = np.nan
raw = raw.sort_values(['symbol', 'date'])
raw['ret'] = raw.groupby('symbol')['close'].pct_change().clip(-.11, .11)
print('原始 %d 行 | %d 只 | %s ~ %s | %.0fs' %
      (len(raw), raw.symbol.nunique(), raw.date.min().date(), raw.date.max().date(),
       time.time() - t0), flush=True)

wide = raw.pivot_table(index='date', columns='symbol', values='ret', aggfunc='last')
days = wide.index
out = []
for year in range(2016, 2027):
    hist = wide.loc[days < f'{year}-01-01'].tail(WIN)
    if len(hist) < 100:
        continue
    cnt = hist.notna().sum(0)
    keep = cnt[cnt >= MINOBS].index
    R = hist[keep].values.astype(np.float32)
    R = np.nan_to_num(R, nan=0.0)
    R = (R - R.mean(0)) / (R.std(0) + 1e-6)
    R = np.clip(R, -5, 5)
    Z = PCA(n_components=min(NPC, R.shape[1] - 1), svd_solver='randomized',
            random_state=0).fit_transform(R.T)          # [N, NPC]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    lab = KMeans(n_clusters=K, n_init=5, random_state=0).fit_predict(Z)
    d = pd.DataFrame({'year': year, 'symbol': keep, 'cluster': lab})
    out.append(d)
    print('  %d 年：用 %d 个历史交易日 (%s~%s)，覆盖 %d 只，%d 组 | %.0fs' %
          (year, len(hist), hist.index[0].date(), hist.index[-1].date(),
           len(keep), len(set(lab)), time.time() - t0), flush=True)

d = pd.concat(out, ignore_index=True)
d.to_parquet(OUT, index=False)
print('\n已存 %s | %d 行 | %d 年份 | %d 只' %
      (OUT, len(d), d.year.nunique(), d.symbol.nunique()))
print('每年分组数:', d.groupby('year')['cluster'].nunique().to_dict())
print('平均组内股票数 %.0f' % d.groupby(['year', 'cluster']).size().mean())
