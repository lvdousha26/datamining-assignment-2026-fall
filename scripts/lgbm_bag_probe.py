# -*- coding: utf-8 -*-
"""TF 的"稳定性优势"是不是只是 bagging 的功劳？

背景：dev 段 TF(3 seed bagging) ICIR 1.21 高于 LGBM(单模型) 1.00，容易被读成
"Transformer 更稳"。但 3 个种子平均本身就会降低预测方差——这是 bagging 的数学性质，
不是结构优势。

本脚本给 LGBM 也做 3 seed bagging（同训练集、同轮数、只改 seed），对比：
    单 seed LGBM  vs  3 seed LGBM  vs  TF 3 seed
若 3 seed LGBM 的 ICIR 也跳到 1.2 附近，则 TF 的稳定性优势完全来自 bagging。
"""
import json
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy import stats as sps

ROOT = r'D:/quant'
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates'])
T, N, F = X.shape
LOCK_START = str(z['lock_start']) if 'lock_start' in z else '2025-07-01'
SEEDS = (42, 7, 2024)

base = dict(objective='regression', metric='l2', learning_rate=0.03, num_leaves=63,
            min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=8)
NROUND = json.load(open(f'{ROOT}/checkpoints/lgbm_meta.json'))['best_iter']

def emb(mask, e=6):
    """与 big_train 一致：每段尾部剔除 e 个交易日，防止 5 日重叠标签跨边界"""
    idx = np.where(mask)[0]
    if len(idx) > e: mask[idx[-e:]] = False
    return mask


tr_days = np.where(emb(dates < '2024-01-01'))[0]
sel = tr_days[np.arange(0, len(tr_days), 5)]
md_tr = np.zeros(T, bool); md_tr[sel] = True
md_te = np.zeros(T, bool)
md_te[np.where((dates >= '2025-01-01') & (dates < LOCK_START))[0]] = True
md_te = emb(md_te)

tt, ii = np.where(M & md_tr[:, None] & np.isfinite(Y))
ta, ia = np.where(M & md_te[:, None] & np.isfinite(Y))


def yrank(t_, i_):
    cache = {}; out = np.empty(len(t_), np.float32)
    for k, (t, i) in enumerate(zip(t_, i_)):
        if t not in cache:
            m = M[t] & np.isfinite(Y[t]); r = np.full(N, np.nan, np.float32)
            if m.sum() >= 30:
                v = Y[t][m]; r[m] = (sps.rankdata(v) - .5) / m.sum()
            cache[t] = r
        out[k] = cache[t][i]
    return out


def dic(pr):
    d = pd.DataFrame({'d': ta, 'p': pr, 'y': Y[ta, ia]})
    s = d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    return s


print('训练 %d 行 / 验证 %d 行（dev 2025H1）| 轮数 %d' % (len(tt), len(ta), NROUND), flush=True)
ytr = yrank(tt, ii)
P = []
for s in SEEDS:
    p = dict(base); p['seed'] = s; p['bagging_seed'] = s; p['feature_fraction_seed'] = s
    m = lgb.train(p, lgb.Dataset(X[tt, ii, :], ytr), num_boost_round=NROUND)
    pr = m.predict(X[ta, ia, :], num_iteration=NROUND)
    sc = dic(pr)
    print('  seed%-5d 单模型 IC %+.4f | ICIR %.2f | IC std %.4f' %
          (s, sc.mean(), sc.mean() / sc.std(), sc.std()), flush=True)
    P.append(pr)

P = np.array(P)
for tag, pr in [('3 seed bagging', P.mean(0)), ('seed42 only', P[0])]:
    sc = dic(pr)
    print('  %-15s IC %+.4f | ICIR %.2f | IC std %.4f' %
          (tag, sc.mean(), sc.mean() / sc.std(), sc.std()))

p_tf = np.load(f'{ROOT}/data/big/preds.npz')['p_tf_te']
s_tf = dic(p_tf)
print('  %-15s IC %+.4f | ICIR %.2f | IC std %.4f  ← 对照' %
      ('TF 3 seed', s_tf.mean(), s_tf.mean() / s_tf.std(), s_tf.std()))

# 逐日差检验：bagging LGBM vs TF bagging
s_bag = dic(P.mean(0))
diff = s_tf.loc[s_bag.index.intersection(s_tf.index)] - s_bag.loc[s_bag.index.intersection(s_tf.index)]


def nw(x, lag=4):
    x = np.asarray(x); n = len(x); e = x - x.mean(); v = e @ e / n
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1); v += 2 * w * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(v / n)


print('\nTF(bag) − LGBM(bag) 逐日差：均值 %+.5f | NW-t %+.2f | TF 胜 %d/%d 日' %
      (diff.mean(), nw(diff), int((diff > 0).sum()), len(diff)))
print('两者预测秩相关：%.3f' % np.mean(
    [sps.spearmanr(P.mean(0)[ta == t], p_tf[ta == t]).correlation for t in np.unique(ta)]))

json.dump({'seeds': list(SEEDS), 'rounds': NROUND,
           'lgb_single_icir': float(dic(P[0]).mean() / dic(P[0]).std()),
           'lgb_bag_ic': float(s_bag.mean()), 'lgb_bag_icir': float(s_bag.mean() / s_bag.std()),
           'tf_bag_ic': float(s_tf.mean()), 'tf_bag_icir': float(s_tf.mean() / s_tf.std()),
           'tf_minus_lgb_bag_nwt': float(nw(diff))},
          open(f'{ROOT}/reports/lgbm_bag_probe.json', 'w'), indent=2, ensure_ascii=False)
print('已存 reports/lgbm_bag_probe.json')
