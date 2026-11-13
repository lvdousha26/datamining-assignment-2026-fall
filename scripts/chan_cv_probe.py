# -*- coding: utf-8 -*-
"""chan_cv 结果的泄漏探针：+zs_pos 的 Δ+0.0153 好得可疑，做三个对照

  ① 列置乱   ZS 的股票维度随机重排（NaN 模式跟着列走）→ 若 Δ 仍在，说明增益不来自个股信息
  ② 滞后 60 日  把 60 天前的 zs_pos 安到今天 → 若 Δ 仍在，说明与"当下状态"无关
  ③ 提前 10 日  把 t+10 的 zs_pos 安到今天（覆盖标签窗 t+1..t+6）→ 若 Δ 与①live 相当，
                说明对齐可能有系统性偏移；若 Δ 明显更大，才是"真用到未来"的量级参照

用法：python scripts/chan_cv_probe.py
产出：追加到 reports/chan_cv.md 的「探针」一节 + 控制台
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
PURGE, EMBARGO, STEP = 6, 6, 5
params = dict(objective='regression', metric='l2', learning_rate=0.03, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=8, seed=42)
NROUND = json.load(open(f'{ROOT}/checkpoints/lgbm_meta.json'))['best_iter']


def yrank(Y, M, tt, ii):
    cache = {}
    out = np.empty(len(tt), np.float32)
    for k, (t, i) in enumerate(zip(tt, ii)):
        if t not in cache:
            m = M[t] & np.isfinite(Y[t]); r = np.full(Y.shape[1], np.nan, np.float32)
            if m.sum() >= 30:
                r[m] = (sps.rankdata(Y[t][m]) - .5) / m.sum()
            cache[t] = r
        out[k] = cache[t][i]
    return out


def daily_ic(p, tt, ii, Y):
    d = pd.DataFrame({'d': tt, 'p': p, 'y': Y[tt, ii]})
    return d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()


def cv_ic(X, Y, M, dates, extra):
    """extra: [T,N] 追加因子；返回 5 折 IC 列表"""
    XA = np.concatenate([X, extra[:, :, None]], axis=2)
    T = X.shape[0]
    usable = np.where((M & np.isfinite(Y)).any(1))[0]
    bounds = np.linspace(usable[0], usable[-1] + 1, 7).astype(int)
    ics = []
    for k in range(5):
        te0, te1 = bounds[k + 1], bounds[k + 2]
        tr0, tr1 = bounds[0], bounds[k + 1] - PURGE
        mtr = np.zeros(T, bool); mtr[tr0:tr1:STEP] = True
        mte = np.zeros(T, bool); mte[te0:te1:STEP] = True
        tt, ii = np.where(M & mtr[:, None] & np.isfinite(Y))
        ta, ia = np.where(M & mte[:, None] & np.isfinite(Y))
        ytr = yrank(Y, M, tt, ii)
        ml = lgb.train(params, lgb.Dataset(XA[tt, ii, :], ytr), num_boost_round=NROUND)
        p = ml.predict(XA[ta, ia, :], num_iteration=NROUND)
        ics.append(daily_ic(p, ta, ia, Y).mean())
    return ics


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, Y, M = z['X'], z['Y2'], z['M']
    dates = pd.to_datetime(z['dates'])
    ZS = np.load(f'{ROOT}/data/big/chan_zspos.npy')

    rng = np.random.default_rng(7)
    probes = {}
    # ① 列置乱
    perm = rng.permutation(ZS.shape[1])
    probes['①列置乱'] = ZS[:, perm]
    # ② 滞后 60 交易日
    st = np.full_like(ZS, np.nan); st[60:] = ZS[:-60]
    probes['②滞后60日'] = st
    # ③ 提前 10 交易日（含标签窗未来信息）
    fu = np.full_like(ZS, np.nan); fu[:-10] = ZS[10:]
    probes['③提前10日'] = fu

    base = cv_ic(X, Y, M, dates, ZS)          # live（应复现 +0.1194）
    print('live    5 折 IC:', [round(v, 4) for v in base], flush=True)
    out = {}
    for name, E in probes.items():
        ics = cv_ic(X, Y, M, dates, E)
        out[name] = ics
        print('%-8s 5 折 IC: %s | Δ %s' % (name, [round(v, 4) for v in ics],
              [round(b - a, 4) for b, a in zip(ics, base)]), flush=True)

    L = ['', '## 泄漏探针（对照实验）', '',
         '| 探针 | 做法 | 5 折 IC 均值 | Δ vs live | 结论 |', '|---|---|---|---|---|']
    judged = {
        '①列置乱': '增益是否来自个股信息（置乱后应归零）',
        '②滞后60日': '增益是否依赖"当下"状态（滞后后应明显衰减）',
        '③提前10日': '未来信息的量级参照（应显著大于 live 才正常）',
    }
    for name, ics in out.items():
        L.append('| %s | %s | %+.4f | %+.4f | %s |' % (
            name, judged[name], np.mean(ics), np.mean(ics) - np.mean(base), ''))
    L += ['']
    open(f'{ROOT}/reports/chan_cv.md', 'a', encoding='utf-8', newline='\n').write('\n'.join(L))
    print('\n→ 已追加到 reports/chan_cv.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='chan_probe'):
        main()
