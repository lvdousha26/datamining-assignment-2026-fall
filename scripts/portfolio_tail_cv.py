# -*- coding: utf-8 -*-
"""组合层面终审：尾部加权值不值钱？（purged CV 协议，看年化超额）

之前的链条：
  IC 口径 → 「尾部加权无效」
  D1 命中率口径 → 「有效（+3.3pp / 全市场 +2.6pp）」
  全市场 D1 桶超额 → 「避损反而变差」

D1 桶均值仍然不是钱。真正的问题是：**剔掉 D1 之后，组合的年化超额变不变**。
本脚本按 `purged_cv.py` 的协议（6 块 expanding、purge 6、每 5 日采样、固定 85 轮），
在每一折的验证段上：

  1. 每期取打分最低 10% 作为 D1；
  2. 组合超额 = 保留部分的 5 日超额（= −D1 组的 5 日超额，Y2 已截面去均值）；
  3. 扣成本：churn × 10% × 2 × 15bp（与 `evaluate_screen.py` 同口径）；
  4. 5 折各自的年化超额 → 配对 t 检验。

**为什么用 LGBM 而不是 NN**：NN 在全市场上跑 5 折 × 2 变体 ≈ 30 次训练（约 3 小时），
而尾部加权是可以直接搬到 LGBM 的样本权重上的。先在生产模型上验，若组合层面有效，
再决定是否为 NN 付那 10 倍算力。

用法：python scripts/portfolio_tail_cv.py
产出：reports/portfolio_tail_cv.md + csv
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from screen_stats import d1_churn  # noqa: E402

ROOT = r'D:/quant'
PURGE, STEP, Q, COST = 6, 5, 0.10, 0.0015
NROUND = 85
PARAMS = dict(objective='regression', metric='l2', learning_rate=0.03, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=8, seed=42)


def yrank(Y, M, tt, ii):
    out = np.empty(len(tt), np.float32)
    for t in np.unique(tt):
        idx = np.where(tt == t)[0]
        v = Y[t, ii[idx]]
        out[idx] = (np.argsort(np.argsort(v)) + 0.5) / len(v)
    return out


def eval_fold(pred, yv, tv, iv):
    """逐期：保留部分的 5 日超额（Y2 已去均值）− 成本；返回年化"""
    d = pd.DataFrame({'t': tv, 'i': iv, 'p': pred, 'y': yv})
    churn = {t: (n, c) for t, n, c in d1_churn(d, q=Q, step=1, min_n=200)}
    rows = []
    for t, (n, to) in churn.items():
        g = d[d.t == t]
        d1 = g.nsmallest(n, 'p')            # D1 是打分最低的，剔掉
        # 保留部分（N-n）相对全市场的超额：keep_mean - all_mean = -n/(N-n) * (d1_mean - all_mean)
        ex = -(n / (len(g) - n)) * (d1.y.mean() - g.y.mean())
        rows.append(ex - to * Q * 2 * COST)
    r = np.array(rows)
    ann = (1 + r).prod() ** (252 / STEP / len(r)) - 1
    return ann, len(r)


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, Y, M = z['X'], z['Y2'], z['M']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    T, N, F = X.shape
    usable = np.where((M & np.isfinite(Y)).any(1))[0]
    bounds = np.linspace(usable[0], usable[-1] + 1, 7).astype(int)
    print('面板 %s | 可用 %d 日 | 折界 %s' % (X.shape, len(usable), bounds.tolist()), flush=True)

    res = {'base': [], 'tail': []}
    for k in range(5):
        te0, te1 = bounds[k + 1], bounds[k + 2]
        tr1 = bounds[k + 1] - PURGE
        mtr = np.zeros(T, bool); mtr[bounds[0]:tr1:STEP] = True
        mte = np.zeros(T, bool); mte[te0:te1:STEP] = True
        tt, ii = np.where(M & mtr[:, None] & np.isfinite(Y))
        ta, ia = np.where(M & mte[:, None] & np.isfinite(Y))
        if len(tt) < 10000 or len(ta) < 10000:
            print('折%d 样本不足，跳过' % (k + 1)); continue
        Xtr = X[tt, ii, :]
        Xva = X[ta, ia, :]
        ytr = yrank(Y, M, tt, ii)
        rk = ytr
        yv = Y[ta, ia]

        for tag, w in [('base', None), ('tail', 1.0 + 4.0 * (rk < 0.1))]:
            t0 = time.time()
            ds = lgb.Dataset(Xtr, ytr, weight=w)
            m = lgb.train(PARAMS, ds, num_boost_round=NROUND)
            p = m.predict(Xva)
            ann, nper = eval_fold(p, yv, ta, ia)
            res[tag].append(ann)
            print('  折%d %-5s 年化超额 %+7.2f%% | %d 期 | %.0fs' % (
                k + 1, tag, ann * 100, nper, time.time() - t0), flush=True)
        del Xtr, Xva

    a, b = np.array(res['base']), np.array(res['tail'])
    t, p = sps.ttest_rel(b, a)
    print('\n5 折均值：base %+.2f%% | tail %+.2f%% | 差 %+.2fpp | 配对 t=%.2f p=%.4f' % (
        a.mean() * 100, b.mean() * 100, (b.mean() - a.mean()) * 100, t, p))

    df = pd.DataFrame({'fold': range(1, len(a) + 1), 'base': a, 'tail': b})
    df.to_csv(f'{ROOT}/reports/portfolio_tail_cv.csv', index=False)
    md = ['# 组合层面终审：尾部加权值不值钱', '',
          '> %s｜purged CV（6 块 expanding / purge %d / 每 %d 日采样 / 固定 %d 轮）｜标签 Y2（5 日超额）' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), PURGE, STEP, NROUND), '',
          '口径：每期剔掉打分最低 10%，组合超额 = 保留部分相对全市场的 5 日超额，扣成本（churn×10%×2×15bp）。', '',
          '| 折 | 无加权 年化超额 | 尾部加权 年化超额 | 差 |', '|---|---|---|---|']
    for _, r in df.iterrows():
        md.append('| %d | %+.2f%% | %+.2f%% | %+.2fpp |' % (
            r['fold'], r['base'] * 100, r['tail'] * 100, (r['tail'] - r['base']) * 100))
    md += ['| **均值** | **%+.2f%%** | **%+.2f%%** | **%+.2fpp** |' % (
        a.mean() * 100, b.mean() * 100, (b.mean() - a.mean()) * 100),
        '', '配对 t 检验：t = %.2f，p = %.4f → %s' % (
            t, p, '差异显著' if p < 0.05 else '**差异不显著**'), '',
        '## 结论', '']
    if p >= 0.05:
        md += ['**尾部加权在组合层面没有带来可测的改善**（p = %.2f）。' % p,
               '结合前两轮的结论，最终判断是：**不改生产**。',
               '命中率提升是真的，但它提到的是「好抓的边缘输家」，对组合年化超额没有贡献。']
    else:
        d = (b.mean() - a.mean()) * 100
        md += ['**尾部加权在组合层面%s**：5 折年化超额 %+.2fpp（p = %.4f）。' % (
            '有效' if d > 0 else '反而更差', d, p),
               '注意方向：%s' % ('正向，可以进生产' if d > 0 else '负向，不要改生产')]
    md += ['', '## 为什么最终以这个指标为准', '',
           'IC、命中率、D1 桶均值都是**代理指标**，它们之间会互相矛盾（前三轮已经各给过一次相反结论）；',
           '组合年化超额是唯一直接对应收益的指标。代理之间打架时，以它为准。', '',
           '## 局限', '',
           '- 只在 LGBM 上验证（NN 的 5 折 × 2 变体约 3 小时，未跑）；',
           '- 5 折样本量小，配对 t 只有 5 个点，功效有限；',
           '- 成本按 15bp 单边、churn×10% 估算，实际冲击成本随规模上升。', '']
    open(f'{ROOT}/reports/portfolio_tail_cv.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('→', f'{ROOT}/reports/portfolio_tail_cv.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='port_tail_cv'):
        main()
