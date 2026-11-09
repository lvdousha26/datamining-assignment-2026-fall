# -*- coding: utf-8 -*-
"""把 arr.npz 的 IND 换成申万二级，并给 panel_big 加 sw_industry 列

策略：
- arr.npz 是 zip 容器，只替换 IND.npy 成员，其余成员流式拷贝（不解压、不重算因子）
- 原聚类 IND 备份为 IND_cluster.npy
- panel_big.parquet 保留原 industry 列，新增 sw_industry（可回退）
"""
import io, json, os, shutil, zipfile
import numpy as np, pandas as pd

ROOT = 'D:/quant'
SRC = f'{ROOT}/data/big/arr.npz'
TMP = f'{ROOT}/data/big/arr_new.npz'

z = np.load(SRC, allow_pickle=True)
IND_old = z['IND'].copy()
IND_sw = np.load(f'{ROOT}/data/big/IND_sw.npy')
names = pd.read_json(f'{ROOT}/data/big/IND_sw_names.json', typ='series')
print('旧 IND: %s 类别 %d | 新 IND: %s 类别 %d' %
      (IND_old.shape, len(np.unique(IND_old[IND_old >= 0])),
       IND_sw.shape, len(np.unique(IND_sw[IND_sw >= 0])), ), flush=True)
assert IND_old.shape == IND_sw.shape

np.save(f'{ROOT}/data/big/IND_cluster.npy', IND_old.astype(np.int8))
print('已备份 IND_cluster.npy', flush=True)

buf = io.BytesIO()
np.save(buf, IND_sw.astype(np.int8))
new_bytes = buf.getvalue()

with zipfile.ZipFile(SRC) as zi, zipfile.ZipFile(TMP, 'w', zipfile.ZIP_STORED, allowZip64=True) as zo:
    for item in zi.infolist():
        if item.filename == 'IND.npy':
            zo.writestr(item.filename, new_bytes)
        else:
            with zi.open(item.filename) as fs, zo.open(item.filename, 'w', force_zip64=True) as fd:
                shutil.copyfileobj(fs, fd, 8 << 20)
        print('  %-16s %8.1f MB' % (item.filename, item.file_size / 1e6), flush=True)

# 校验后再替换
chk = np.load(TMP, allow_pickle=True)
assert set(chk.files) == set(z.files), '成员不一致'
assert (chk['IND'] == IND_sw.astype(np.int8)).all(), 'IND 未生效'
assert chk['X'].shape == z['X'].shape and chk['Y2'].shape == z['Y2'].shape
assert np.allclose(chk['X'][0, :50].astype(float), z['X'][0, :50].astype(float), equal_nan=True)
chk.close()
os.replace(TMP, SRC)
print('\narr.npz 已替换（IND → 申万二级 %d 类）' % len(np.unique(IND_sw[IND_sw >= 0])))

# panel_big 加列
P = f'{ROOT}/data/big/panel_big.parquet'
pf = pd.read_parquet(P)
dts = pd.to_datetime(pd.Series(z['dates']).astype(str))
syms = [str(s).zfill(6) for s in z['symbols']]
ti = pd.Index(dts).get_indexer(pd.DatetimeIndex(pf['date']))
si = pd.Index(syms).get_indexer(pf['symbol'].astype(str))
ok = (ti >= 0) & (si >= 0)
sw_val = np.full(len(pf), -1, dtype=np.int16)
sw_val[ok] = IND_sw[ti[ok], si[ok]]
pf['sw_industry'] = sw_val
pf.to_parquet(P, index=False)
print('panel_big.parquet 已加 sw_industry（保留原 industry=%d 类）| 覆盖 %.1f%%' %
      (pf['industry'].nunique(), (pf['sw_industry'] >= 0).mean() * 100))
