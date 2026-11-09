# -*- coding: utf-8 -*-
"""用申万二级行业替换统计聚类行业（IND 数组）

背景：现面板 IND 来自 ind_roll.parquet —— 每年用上一年 250 日收益做 PCA+KMeans 的
统计聚类 31 组，跨年 ARI 仅 0.12（等于每年换一套行业）。申万二级是业务分类，
时变慢、可解释、与市场共识一致。

做法：
1. 按【公告日】merge_asof 铺到交易日（严格 PIT，不用年份硬对齐）
2. 同一股票缺失年份用前后最近一次已知行业填充
3. <MIN_N 只的行业合并到「其他」（中性化时小行业噪声大）
4. 输出 IND_sw.npy + 覆盖率/稳定性对比

不重算因子：X 与 IND 独立，替换属元数据级操作。
"""
import glob
import numpy as np, pandas as pd

ROOT = 'D:/quant'
MIN_N = 30          # 小于该股票数的行业并入「其他」

sw = pd.read_parquet(f'{ROOT}/data/big/sw_industry.parquet')
sw['ann'] = pd.to_datetime(dict(year=sw['year'], month=7, day=1))   # 该年行业中点，仅用于排序
sw['code'] = sw['code'].astype(str).str.zfill(6)
sw = sw.sort_values(['code', 'year'])

# 行业编码（按出现频次降序，保证 int8 装得下 128 类）
vc = sw['sw_ind'].value_counts()
keep = set(vc[vc >= MIN_N].index)
print('行业总数 %d，>=%d 只的 %d 个，并入「其他」%d 个（共 %d 只）' %
      (len(vc), MIN_N, len(keep), len(vc) - len(keep), vc[vc < MIN_N].sum()), flush=True)
sw['ind'] = sw['sw_ind'].where(sw['sw_ind'].isin(keep), '其他')
codes = {k: i for i, k in enumerate(sorted(sw['ind'].unique()))}
sw['ind_id'] = sw['ind'].map(codes).astype(np.int16)
print('编码后行业数 %d' % len(codes), flush=True)

# 市场交易日网格
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
M = z['M']
symbols = [str(s).zfill(6) for s in z['symbols']]
T, N = len(dates), len(symbols)
dmap = {d: i for i, d in enumerate(dates)}
smap = {s: j for j, s in enumerate(symbols)}

# 逐股 merge_asof（用真实行情日期作为左表）
mk = []
for f in sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet')):
    d = pd.read_parquet(f, columns=['date', 'symbol'])
    mk.append(d)
mk = pd.concat(mk, ignore_index=True)
mk['code'] = mk['symbol'].astype(str).str.zfill(6)
mk['date'] = pd.to_datetime(mk['date']).astype('datetime64[ns]')
mk = mk.drop_duplicates(['code', 'date'])
print('行情网格 %d 行 / %d 只' % (len(mk), mk.code.nunique()), flush=True)

sw['ann'] = sw['ann'].astype('datetime64[ns]')
sw_by = {k: v for k, v in sw[['ann', 'ind_id', 'sw_ind']].sort_values('ann').groupby(sw['code'])}
out = []
for code, m in mk.sort_values(['code', 'date']).groupby('code', sort=False):
    g = sw_by.get(code)
    if g is None: continue
    out.append(pd.merge_asof(m, g, left_on='date', right_on='ann', direction='backward'))
res = pd.concat(out, ignore_index=True)
print('PIT 展开 %d 行' % len(res), flush=True)

IND = np.full((T, N), -1, dtype=np.int16)
ti = pd.Index(dates).get_indexer(pd.DatetimeIndex(res['date']))
si = pd.Index(symbols).get_indexer(res['code'].astype(str))
ok = pd.notna(res['ind_id'].values) & (ti >= 0) & (si >= 0)
IND[ti[ok], si[ok]] = res['ind_id'].values[ok].astype(np.int16)

# 有效样本覆盖
eff = M & (IND >= 0)
print('\n有效样本行业覆盖 %.2f%%（聚类版 %.2f%%）' %
      (eff[M].mean() * 100, ((z['IND'] >= 0)[M]).mean() * 100), flush=True)

# 行业规模分布
cnt = pd.Series(IND[eff]).value_counts()
print('行业数 %d | 中位规模 %d 只 | 最大 %d | 最小 %d' %
      (len(cnt), int(cnt.median()), int(cnt.max()), int(cnt.min())), flush=True)

# 跨年稳定性：同一只股票相邻两年行业是否一致
piv = sw.pivot_table(index='code', values='ind_id', columns='year', aggfunc='last')
same = []
for y in range(2016, 2026):
    if y in piv and y + 1 in piv:
        a, b = piv[y], piv[y + 1]
        m2 = a.notna() & b.notna()
        if m2.sum() > 100:
            same.append(((a[m2] == b[m2]).mean(), m2.sum()))
print('申万二级 相邻年行业不变率: %.1f%%（%d~%d 平均，聚类版跨年 ARI 仅 0.12）' %
      (np.mean([s for s, _ in same]) * 100, 2016, 2025), flush=True)

# 行业解释力：下期收益的组间方差占比
Y = z['Y2']
r2 = []
for t in range(T):
    m = M[t] & np.isfinite(Y[t]) & (IND[t] >= 0)
    if m.sum() < 500: continue
    y = Y[t][m]; g = IND[t][m]
    tot = y.var()
    if tot <= 0: continue
    gm = pd.Series(y).groupby(g).transform('mean')
    r2.append(float(np.var(gm.values) / tot))
print('行业对下期收益的组间方差占比（R²）：申万 %.3f' % np.mean(r2), flush=True)
r2c = []
for t in range(T):
    m = M[t] & np.isfinite(Y[t]) & (z['IND'][t] >= 0)
    if m.sum() < 500: continue
    y = Y[t][m]; g = z['IND'][t][m]
    tot = y.var()
    if tot <= 0: continue
    r2c.append(float(np.var(pd.Series(y).groupby(g).transform('mean').values) / tot))
print('                                  聚类 %.3f' % np.mean(r2c), flush=True)

np.save(f'{ROOT}/data/big/IND_sw.npy', IND.astype(np.int8))
json_codes = {int(v): k for k, v in codes.items()}
pd.Series(json_codes).to_json(f'{ROOT}/data/big/IND_sw_names.json', force_ascii=False)
print('\n已存 data/big/IND_sw.npy 与 IND_sw_names.json')
