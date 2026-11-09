# -*- coding: utf-8 -*-
"""月度避雷清单生成器（模型完全冻结，不重训、不调参）

用法：
  python scripts/live_screen.py                 # 用当前 arr.npz 的最后一个交易日打分
  python scripts/live_screen.py --top 50       # 同时在终端打印 D1 前 50 只

数据更新（每月末）：
  1) python scripts/big_download.py            # 全市场增量下载（分片断点续传）
  2) python scripts/big_features.py            # 重建面板（会写新的 arr.npz / lockbox_labels.npz）
  3) python scripts/live_screen.py             # 生成当期清单
注意：模型权重与 Stacking 系数在 checkpoints/ 与 reports/big_models_meta.json 中冻结，
      重建面板不会改变模型；lockbox 已验证过一次，不要再重复评估。
"""
import os, sys, json
import numpy as np, pandas as pd
import torch, lightgbm as lgb
from tf_net import TFNet, L

ROOT = r'D:/quant'
OUT = f'{ROOT}/live'
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TOPN = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 30

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, M = z['X'], z['M']
dates = pd.to_datetime(z['dates']); syms = np.array([str(s) for s in z['symbols']])
FEATS = list(z['feats']); T, N, F = X.shape
t0 = T - 1
ii = np.where(M[t0])[0]
print('打分日 %s | 有效股票 %d 只 | 面板 %d 日' % (dates[t0].date(), len(ii), T), flush=True)

meta = json.load(open(f'{ROOT}/reports/big_models_meta.json'))
coef = np.array(meta['coef'])
wgt = np.array([meta['weights'].get(str(f), 0.0) for f in meta['feature_order']])
lmeta = json.load(open(f'{ROOT}/checkpoints/lgbm_meta.json'))

# Stacking 标准化参数必须与 coef 严格配套：优先用 meta 里冻结的 coef_mu/coef_sd
pm = np.load(f'{ROOT}/data/big/preds.npz')
if 'coef_mu' in meta and 'coef_sd' in meta:
    MU = meta['coef_mu'][:3]
    SD = meta['coef_sd'][:3]
    print('Stacking 参数来源：meta 冻结值（Ridge）', flush=True)
else:
    MU = [pm['p_lgb_me'].mean(), pm['p_tf_me'].mean(), pm['fac_me'].mean()]
    SD = [pm['p_lgb_me'].std(), pm['p_tf_me'].std(), pm['fac_me'].std()]
    print('Stacking 参数来源：preds.npz 重算（OLS 旧口径）', flush=True)
print('Stacking 标准参数 mu=%s sd=%s' % (np.round(MU, 4), np.round(SD, 4)), flush=True)

# 1) LightGBM
p_lgb = lgb.Booster(model_file=f'{ROOT}/checkpoints/lgbm_big.txt').predict(
    X[t0, ii, :], num_iteration=lmeta['best_iter'])

# 2) Transformer 3 seed bagging（过去 L 日序列）
tt = np.full(len(ii), t0, dtype=int)
idx = tt[:, None] - np.arange(L - 1, -1, -1)[None, :]
seq = X[idx, ii[:, None], :].astype(np.float32)
xb = torch.from_numpy(seq).to(DEV)
p_tf = np.zeros(len(ii), dtype=np.float64)
for s in (1, 2, 3):
    m = TFNet(F).to(DEV)
    m.load_state_dict(torch.load(f'{ROOT}/checkpoints/tf_seed{s}.pt',
                                 map_location=DEV, weights_only=True))
    m.eval()
    with torch.no_grad():
        p_tf += m(xb).cpu().numpy() / 3
print('LGBM / TF 推理完成', flush=True)

# 3) 因子合成（当日截面 rank @ ICIR 权重）
blk = X[t0]
r = np.full((N, F), np.nan, np.float32)
sub = blk[M[t0]]
r[M[t0]] = ((sub.argsort(0).argsort(0) + .5) / len(sub)).astype(np.float32)
fac = (np.nan_to_num(r) @ wgt)[ii]

# 4) Stacking
A = np.column_stack([(p_lgb - MU[0]) / SD[0], (p_tf - MU[1]) / SD[1],
                     (fac - MU[2]) / SD[2], np.ones(len(ii))])
score = A @ coef

# ---- 组装清单 ----
d = pd.DataFrame({'symbol': syms[ii], 'score': score})
d['pct'] = d['score'].rank(pct=True)
d['decile'] = pd.qcut(d['pct'], 10, labels=False, duplicates='drop') + 1
d['is_d1'] = (d['pct'] <= .10).astype(int)
# 行业（point-in-time 滚动聚类，取当年）
try:
    ind = pd.read_parquet(f'{ROOT}/data/big/ind_roll.parquet')
    ind = ind[ind['year'] == int(dates[t0].year)][['symbol', 'cluster']]
    ind['symbol'] = ind['symbol'].astype(str)
    d = d.merge(ind, on='symbol', how='left').rename(columns={'cluster': 'industry'})
except Exception:
    d['industry'] = np.nan
# 名称：本地缓存，避免每次月度运行都联网拉全市场快照
nmfile = f'{ROOT}/data/big/name_map.parquet'
try:
    if os.path.exists(nmfile):
        nm = pd.read_parquet(nmfile)
    else:
        import akshare as ak, contextlib, io
        with contextlib.redirect_stderr(io.StringIO()):   # 屏蔽东财接口的 tqdm 进度条
            nm = ak.stock_zh_a_spot_em()[['代码', '名称']]
        nm.columns = ['symbol', 'name']
        nm['symbol'] = nm['symbol'].astype(str).str.zfill(6)
        nm.to_parquet(nmfile, index=False)
    d = d.merge(nm, on='symbol', how='left')
except Exception:
    d['name'] = ''
d.insert(0, 'date', str(dates[t0].date()))
d = d.sort_values('score').reset_index(drop=True)
d['d1_rank'] = d['is_d1'].cumsum().where(d['is_d1'] == 1)

path = f'{OUT}/screen_{dates[t0].date()}.csv'
d.to_csv(path, index=False, float_format='%.6f')
print('\n已存 %s' % path)
print('D1（垫底 10%%）= %d 只 | 分数最低 5 只：' % d.is_d1.sum())
print(d.head(5)[['symbol', 'name', 'score', 'pct', 'industry']].to_string(index=False))
print('\nD1 前 %d 只（禁买名单）：' % TOPN)
print(d.head(TOPN)[['symbol', 'name', 'pct', 'industry']].to_string(index=False))
print('\nD10（最高分组）%d 只 —— 参考，不建议据此选股（小盘冷门股，成本高）' %
      (d.pct > .9).sum())
