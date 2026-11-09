# -*- coding: utf-8 -*-
"""重建前后对比：断层清洗对标签与样本的影响"""
import numpy as np, pandas as pd

ROOT = 'D:/quant'
old = np.load(f'{ROOT}/data/big/arr_backup.npz', allow_pickle=True)
new = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)

print('%-10s %10s %10s' % ('', '旧', '新'))
for k in ['X', 'Y2', 'M', 'IND']:
    print('%-10s %10s %10s' % (k, old[k].shape, new[k].shape))

od = pd.to_datetime(pd.Series(old['dates']).astype(str))
nd = pd.to_datetime(pd.Series(new['dates']).astype(str))
os_ = [str(s).zfill(6) for s in old['symbols']]
ns_ = [str(s).zfill(6) for s in new['symbols']]
print('日期 %s ~ %s (%d) → %s ~ %s (%d)'
      % (od.iloc[0].date(), od.iloc[-1].date(), len(od), nd.iloc[0].date(), nd.iloc[-1].date(), len(nd)))
print('股票 %d → %d | 因子 %d → %d' % (len(os_), len(ns_), old['X'].shape[2], new['X'].shape[2]))

# 对齐：以新网格为基准，找每个新日期/新股票在旧网格里的位置
ot = pd.Index(od).get_indexer(nd)          # 新→旧 的日期位置
osx = pd.Index(os_).get_indexer(ns_)       # 新→旧 的股票位置
ct = np.where(ot >= 0)[0]; cs = np.where(osx >= 0)[0]
print('\n共同日期 %d | 共同股票 %d' % (len(ct), len(cs)))
oT, oS = ot[ct], osx[cs]

O2 = old['Y2'][np.ix_(oT, oS)]; N2 = new['Y2'][np.ix_(ct, cs)]
OM = old['M'][np.ix_(oT, oS)];  NM = new['M'][np.ix_(ct, cs)]

fo, fn = np.isfinite(O2), np.isfinite(N2)
print('\n=== 标签变化 ===')
print('  旧有标签 %d | 新有标签 %d | 净减少 %d (%.2f%%)'
      % (fo.sum(), fn.sum(), fo.sum() - fn.sum(), (fo.sum() - fn.sum()) / fo.sum() * 100))
lost = fo & ~fn
print('  被剔除的标签 %d 条' % lost.sum())
if lost.sum():
    v = O2[lost]
    print('  被剔除标签的旧值：均值 %+.3f | 标准差 %.3f | 区间 [%.1f, %.1f]'
          % (v.mean(), v.std(), v.min(), v.max()))
    print('  其中 |值|>0.5 的占 %.1f%%（正常样本只有 0.01%%）' % ((np.abs(v) > 0.5).mean() * 100))
    print('  按年：' + ' '.join('%d:%d' % (y, n) for y, n in
                               pd.Series(lost.sum(1), index=nd[ct]).groupby(
                                   nd[ct].dt.year).sum().items()))
print('  新样本中 |Y2|>0.5 的占比 %.4f%%（旧 %.4f%%）'
      % ((np.abs(N2[fn]) > 0.5).mean() * 100, (np.abs(O2[fo]) > 0.5).mean() * 100))
print('  新样本标准差 %.4f（旧 %.4f）' % (N2[fn].std(), O2[fo].std()))

print('\n=== 样本掩码 ===')
print('  旧 M %d | 新 M %d | 新增 %d | 减少 %d' % (OM.sum(), NM.sum(), (NM & ~OM).sum(), (OM & ~NM).sum()))

# 行业
oi = old['IND'][np.ix_(oT, oS)]; ni = new['IND'][np.ix_(ct, cs)]
print('\n=== 行业 ===')
print('  旧 %d 类 | 新 %d 类' % (len(np.unique(oi[oi >= 0])), len(np.unique(ni[ni >= 0]))))
print('  同位置一致率 %.1f%%' % ((oi == ni).mean() * 100))
