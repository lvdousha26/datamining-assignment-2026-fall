# -*- coding: utf-8 -*-
"""只重建 Stacking 元模型层（不重训 LGBM / TF）

背景：checkpoint 全在（lgbm_big.txt + tf_seed{1,2,3}.pt），产线仍需 OLS 系数。
重跑 big_train.py 的唯一目的是换元模型，这件事不需要 GPU。

流程：
1. 用训练段(<2024, 抽样 1/5)的 ICIR 权重构造因子合成 fac —— 与 big_train 完全一致
2. 加载 LGBM / 3 个 TF seed，在 meta 段(2024)生成预测
3. stack() 固定 meta 段 mu/sd → 选 lam → 拟合 Ridge
4. 应用到 dev 段(2025H1)，与 OLS 对照
5. 写回 reports/big_models_meta.json（含 coef_mu / coef_sd，供 lockbox_eval 读取）

不碰 lockbox（runs=1 已消耗）。
"""
import json
import numpy as np, pandas as pd
import lightgbm as lgb
import torch
from scipy import stats as sps

import sys
sys.path.insert(0, 'D:/quant/scripts')
from stack_ridge import stack
from tf_net import TFNet

ROOT = 'D:/quant'
L, D, HEADS, LAYERS = 40, 96, 4, 2
EMBARGO = 6
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEV, flush=True)

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M, dates = z['X'], z['Y2'], z['M'], pd.to_datetime(z['dates'])
FEATS = list(z['feats'])
T, N, F = X.shape
LOCK_START = str(z['lock_start']) if 'lock_start' in z else '2025-07-01'


def embargo(mask, e=EMBARGO):
    out = mask.copy(); idx = np.where(out)[0]
    if len(idx) > e: out[idx[-e:]] = False
    return out


dm = {'tr': dates < '2024-01-01',
      'me': (dates >= '2024-01-01') & (dates < '2025-01-01'),
      'va': (dates >= '2025-01-01') & (dates < LOCK_START)}
dm = {k: embargo(v) for k, v in dm.items()}


def samples(md):
    tt, ii = np.where(M & md[:, None] & np.isfinite(Y))
    k = tt >= L
    return tt[k], ii[k]


tr_days = np.where(dm['tr'])[0]
md_tr_w = np.zeros(T, bool); md_tr_w[tr_days[np.arange(0, len(tr_days), 5)]] = True
tt_tr, ii_tr = samples(md_tr_w)
tt_me, ii_me = samples(dm['me'])
tt_va, ii_va = samples(dm['va'])
print('训练 %d / meta %d / dev %d 行' % (len(tt_tr), len(tt_me), len(tt_va)), flush=True)

# ---------- fac：训练段 ICIR 权重 ----------
def rank_mat(blk, m):
    r = np.full_like(blk, np.nan, dtype=np.float32)
    if m.sum() < 30: return r
    sub = blk[m]
    r[m] = ((sub.argsort(0).argsort(0) + 0.5) / len(sub)).astype(np.float32)
    return r


ic_sum = np.zeros(F); ic_cnt = np.zeros(F)
for t in np.where(md_tr_w)[0]:
    m = M[t] & np.isfinite(Y[t])
    if m.sum() < 30: continue
    rk = rank_mat(X[t], m); yv = Y[t][m]
    for k in range(F):
        v = rk[m, k]
        if np.std(v) > 0:
            c = sps.spearmanr(v, yv).correlation
            if np.isfinite(c): ic_sum[k] += c; ic_cnt[k] += 1
icf = ic_sum / np.maximum(ic_cnt, 1)
wgt = np.clip(icf, 0, None); wgt = wgt / wgt.sum()
print('因子权重 top4:', {FEATS[i]: round(float(wgt[i]), 3) for i in np.argsort(-wgt)[:4]}, flush=True)

fac = np.full((T, N), np.nan, dtype=np.float32)
for t in range(T):
    m = M[t]
    if m.sum() < 30: continue
    fac[t] = np.nan_to_num(rank_mat(X[t], m)) @ wgt
fac[~M] = np.nan

# ---------- LGBM 预测 ----------
ml = lgb.Booster(model_file=f'{ROOT}/checkpoints/lgbm_big.txt')
p_lgb_me = ml.predict(X[tt_me, ii_me])
p_lgb_va = ml.predict(X[tt_va, ii_va])
print('LGBM 预测完成', flush=True)

# ---------- TF 预测（3 seed bagging） ----------
def make_seq(tt, ii):
    idx = tt[:, None] - np.arange(L - 1, -1, -1)[None, :]
    return X[idx, ii[:, None], :]


def pred_tf(model, tt, ii, bs=4096):
    out = np.empty(len(tt), dtype=np.float32); model.eval()
    with torch.no_grad():
        for s in range(0, len(tt), bs):
            b = torch.from_numpy(make_seq(tt[s:s + bs], ii[s:s + bs])).to(DEV)
            out[s:s + bs] = model(b).cpu().numpy()
    return out


tfs = []
for s in [1, 2, 3]:
    net = TFNet(F, D, HEADS, LAYERS).to(DEV)
    net.load_state_dict(torch.load(f'{ROOT}/checkpoints/tf_seed{s}.pt', map_location=DEV))
    tfs.append(net)
    print('  tf_seed%d 已载入' % s, flush=True)

p_tf_me = np.mean([pred_tf(m, tt_me, ii_me) for m in tfs], 0)
p_tf_va = np.mean([pred_tf(m, tt_va, ii_va) for m in tfs], 0)
print('TF 预测完成', flush=True)


# ---------- 评估 ----------
def ic_of(sc, tt, ii):
    d = pd.DataFrame({'t': tt, 'p': sc, 'y': Y[tt, ii]})
    s = d.groupby('t').apply(lambda g: sps.spearmanr(g.p, g.y).correlation if len(g) > 30 else np.nan,
                             include_groups=False).dropna()
    return s.mean(), s.mean() / s.std(), len(s)


meta_in = (p_lgb_me, p_tf_me, fac[tt_me, ii_me])
dev_in = (p_lgb_va, p_tf_va, fac[tt_va, ii_va])

s_ridge, info = stack(meta_in, Y[tt_me, ii_me], tt_me, dev_in)
s_ols, _ = stack(meta_in, Y[tt_me, ii_me], tt_me, dev_in, lam=0)

print('\n=== dev 段对照 ===')
for nm, sc in [('LGBM 单模型', p_lgb_va), ('TF 单模型', p_tf_va),
               ('因子合成', fac[tt_va, ii_va]), ('OLS 元模型(旧)', s_ols), ('Ridge 元模型(新)', s_ridge)]:
    m, ir, n = ic_of(sc, tt_va, ii_va)
    print('  %-18s IC %+.4f  ICIR %.2f  (%d 日)' % (nm, m, ir, n))
print('  Ridge lam=%g  系数 %s' % (info['lam'], np.round(info['coef'], 4)))
print('  OLS 系数 %s' % np.round(stack(meta_in, Y[tt_me, ii_me], tt_me, dev_in, lam=0)[1]['coef'], 4))

# 逐日打分秩相关（部署口径：清单会不会变）
d_ols = pd.DataFrame({'t': tt_va, 'a': s_ols})
d_rg = pd.DataFrame({'t': tt_va, 'b': s_ridge})
cc = d_ols.join(d_rg, rsuffix='_').groupby('t').apply(
    lambda g: sps.spearmanr(g.a, g.b).correlation, include_groups=False)
print('  OLS vs Ridge 逐日秩相关：均值 %.4f / 最低 %.4f' % (cc.mean(), cc.min()))

# ---------- 写回产线 meta ----------
old = json.load(open(f'{ROOT}/reports/big_models_meta.json'))
new = dict(old)
new.update(dict(
    coef=[float(v) for v in info['coef']],
    coef_mu=[float(v) for v in info['mu']],
    coef_sd=[float(v) for v in info['sd']],
    # 因子权重必须与本次训练 fac 用的完全一致，否则线上 fac 与训练分布漂移
    weights={FEATS[i]: float(wgt[i]) for i in range(F)},
    factor_ic={FEATS[i]: float(icf[i]) for i in range(F)},
    ridge_lambda=float(info['lam']),
    meta='Ridge（meta 段固定标准化参数，lam 由段内 CV 选择）',
    updated='2026-11-06 rebuild_ridge.py（未重训 LGBM/TF，仅换元模型 + 重算因子权重）',
    dev_ic_ridge=float(ic_of(s_ridge, tt_va, ii_va)[0]),
    dev_ic_ols=float(ic_of(s_ols, tt_va, ii_va)[0]),
))
json.dump(new, open(f'{ROOT}/reports/big_models_meta.json', 'w'), indent=2, ensure_ascii=False)
print('\n已写回 reports/big_models_meta.json（含 coef_mu/coef_sd）')
