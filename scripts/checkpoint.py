# -*- coding: utf-8 -*-
"""断点：保存当前可恢复的训练状态
已完成的：数据面板、因子权重、LightGBM（可确定性复现，6 秒重训）、配置
未完成：Transformer 3 种子（进程在预测阶段被终止，内存模型丢失）、Stacking 元模型
"""
import json, os, time
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy import stats as sps

ROOT = r'D:/quant'
CK = f'{ROOT}/checkpoints'
os.makedirs(CK, exist_ok=True)
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates']); FEATS = list(z['feats'])
T, N, F = X.shape
yr = dates.year
print('面板 %s | %d 只 | %s ~ %s' % (X.shape, N, dates[0].date(), dates[-1].date()), flush=True)

def samples(md):
    tt, ii = np.where(M & md[:, None] & ~np.isnan(Y))
    k = tt >= 40
    return tt[k], ii[k]

trd = np.where(yr < 2022)[0]
md = np.zeros(T, bool); md[trd[np.arange(0, len(trd), 5)]] = True
tt, ii = samples(md)
print('训练样本 %d' % len(tt), flush=True)

def yrank(tt, ii):
    cache = {}
    out = np.empty(len(tt), dtype=np.float32)
    for k, (t, i) in enumerate(zip(tt, ii)):
        if t not in cache:
            m = M[t] & ~np.isnan(Y[t]); r = np.full(N, np.nan, dtype=np.float32)
            if m.sum() >= 30:
                v = Y[t][m]; r[m] = (sps.rankdata(v) - 0.5) / m.sum()
            cache[t] = r
        out[k] = cache[t][i]
    return out

ytr = yrank(tt, ii)
params = dict(objective='regression', metric='l2', learning_rate=0.03, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbosity=-1, num_threads=8, seed=42, deterministic=True)
t0 = time.time()
m = lgb.train(params, lgb.Dataset(X[tt, ii, :], ytr), num_boost_round=400)
m.save_model(f'{CK}/lgbm_big.txt')
print('LGBM 已保存 %d 轮 %.0fs' % (m.current_iteration(), time.time() - t0), flush=True)

# 因子 IC 权重（与训练脚本同口径）
print('计算因子权重...', flush=True)
ics = np.zeros(F); cnt = np.zeros(F)
for t in trd:
    msk = M[t] & ~np.isnan(Y[t])
    if msk.sum() < 30: continue
    blk = X[t][msk]; rk = (blk.argsort(0).argsort(0) + .5) / len(blk)
    for k in range(F):
        c = sps.spearmanr(rk[:, k], Y[t][msk]).correlation
        if np.isfinite(c): ics[k] += c; cnt[k] += 1
ic_per_f = ics / np.maximum(cnt, 1)
w = np.clip(ic_per_f, 0, None); w = w / w.sum()

cfg = dict(created=str(pd.Timestamp.now()), panel='data/big/arr.npz',
           label='Y2: t日收盘出信号 → t+1收盘买入 → 持有5日（可执行口径）',
           shape=list(X.shape), n_stocks=int(N), n_days=int(T),
           features=FEATS, splits=dict(train='<2022', meta='2022', val='2023', test='>=2024'),
           lgbm_params=params, lgbm_rounds=400,
           tf=dict(seq_len=40, d_model=96, heads=4, layers=2, dropout=0.15,
                   opt='AdamW', lr=8e-4, wd=1e-2, epochs=6, seeds=[1, 2, 3]),
           factor_ic={FEATS[i]: float(ic_per_f[i]) for i in range(F)},
           factor_weight={FEATS[i]: float(w[i]) for i in range(F)})
json.dump(cfg, open(f'{CK}/config.json', 'w'), ensure_ascii=False, indent=2)

man = {
    'done': ['原始数据 5563 只', '因子面板 5378 只 × 2616 日 × 42 因子（993 万有效样本）',
             'LightGBM 模型文件', '因子 IC/权重', '配置'],
    'lost': ['Transformer 3 个种子（进程在预测阶段被终止）', 'Stacking 元模型', '本轮预测文件'],
    'resume_hint': 'python scripts/big_train.py（已加断点保存，TF 每训练完一个种子即落盘）',
    'known_results_prior_run': {
        'LGBM_OOS_IC': 0.0965, 'TF_OOS_IC': '约 0.10（上一轮，同日标签口径）',
        'note': '上一轮用同日标签，IC 0.1026；改用次日买入后 0.0965'
    }
}
json.dump(man, open(f'{CK}/manifest.json', 'w'), ensure_ascii=False, indent=2)
print('权重 top5:', {FEATS[i]: round(float(w[i]), 3) for i in np.argsort(-w)[:5]}, flush=True)
print('checkpoint 已写入', CK, flush=True)
