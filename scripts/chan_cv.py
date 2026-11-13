# -*- coding: utf-8 -*-
"""缠论 zs_pos 的 5 折 Purged CV 增量检验（项目硬规矩：加因子必须过 CV 增量）

设计
----
复用 `purged_cv.py` 的协议，一字不改：
  - 只在有标签区间切 6 块，折 k 用 B0..Bk 训练（expanding），B(k+1) 验证
  - purge 6 日 / embargo 6 日 / STEP=5 抽样 / 固定 NROUND=85（正式模型 best_iter）不逐折选参
  - LGBM 参数与正式模型一致

三臂对比（同一折同一验证段）：
  A  基线      X（42 因子）
  B  +原始     X + chan_zs_pos（全市场逐只计算，因果修正后）
  C  +残差     X + chan_zs_pos 残差（逐日截面回归剔除 momq60/ret60，不用标签、无泄漏）

判定标准：B/C 相对 A 的逐折 IC 差——5 折里至少 4 折不下降、均值上升才算有增量。

用法：python scripts/chan_cv.py
产出：reports/chan_cv.md + chan_cv_folds.csv
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
import chan_factors as cf  # noqa: E402

ROOT = r'D:/quant'
PURGE, EMBARGO, STEP = 6, 6, 5

params = dict(objective='regression', metric='l2', learning_rate=0.03, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=8, seed=42)
NROUND = json.load(open(f'{ROOT}/checkpoints/lgbm_meta.json'))['best_iter']


def build_zspos():
    """全市场逐只算 chan_zs_pos，返回 [T, N]（对齐 arr 面板）"""
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    dates = pd.to_datetime(z['dates'])
    syms = [str(s).zfill(6) for s in z['symbols']]
    T, N = len(dates), len(syms)
    out = np.full((T, N), np.nan, np.float32)
    sym_idx = {s: i for i, s in enumerate(syms)}
    t0 = time.time()
    n_ok, n_skip = 0, 0
    for k, s in enumerate(syms):
        f = f'{ROOT}/data/big/raw_hfq/{s}.parquet'
        if not os.path.exists(f):
            n_skip += 1
            continue
        d = pd.read_parquet(f, columns=['date', 'open', 'high', 'low', 'close'])
        if len(d) < 60:
            n_skip += 1
            continue
        F = cf.one_stock(d['open'].values, d['high'].values,
                         d['low'].values, d['close'].values)
        v = pd.Series(F['chan_zs_pos'], index=pd.to_datetime(d['date'])).reindex(dates).values
        out[:, sym_idx[s]] = v
        n_ok += 1
        if (k + 1) % 800 == 0:
            print('  zs_pos %d/%d 只（%.0fs）' % (k + 1, len(syms), time.time() - t0), flush=True)
    print('  zs_pos 完成：%d 只入库 / %d 只跳过 / %.0fs / 覆盖率 %.1f%%' %
          (n_ok, n_skip, time.time() - t0, 100 * np.isfinite(out).mean()), flush=True)
    return out


def resid_zspos(X, ZS, feats, tr_days=None):
    """逐日截面回归剔除 momq60 + ret60（不用标签，无泄漏），返回残差矩阵"""
    j1, j2 = feats.index('momq60'), feats.index('ret60')
    out = np.full_like(ZS, np.nan, np.float32)
    days = range(ZS.shape[0]) if tr_days is None else tr_days
    for t in days:
        m = np.isfinite(ZS[t]) & np.isfinite(X[t, :, j1]) & np.isfinite(X[t, :, j2])
        if m.sum() < 300:
            continue
        A = np.column_stack([np.ones(m.sum()), X[t, m, j1], X[t, m, j2]])
        beta, *_ = np.linalg.lstsq(A, ZS[t, m].astype(np.float64), rcond=None)
        out[t, m] = (ZS[t, m] - A @ beta).astype(np.float32)
    return out


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


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, Y, M = z['X'], z['Y2'], z['M']
    dates = pd.to_datetime(z['dates'])
    FEATS = list(z['feats'])
    T, N, F = X.shape

    ZS = build_zspos()
    np.save(f'{ROOT}/data/big/chan_zspos.npy', ZS)
    RS = resid_zspos(X, ZS, FEATS)
    print('  残差覆盖率 %.1f%%' % (100 * np.isfinite(RS).mean()), flush=True)

    # X 逐列 rank（与面板存储口径一致：树喂原 X，本脚本 B/C 臂用同一 X 再拼一列）
    usable = np.where((M & np.isfinite(Y)).any(1))[0]
    print('可用日期 %d 个：%s ~ %s | NROUND=%d' %
          (len(usable), dates[usable[0]].date(), dates[usable[-1]].date(), NROUND), flush=True)
    bounds = np.linspace(usable[0], usable[-1] + 1, 7).astype(int)

    def arm(name):
        """惰性构建：每次调用生成一份拼接矩阵，用完即释放，峰值内存只有一份拷贝"""
        if name == 'A 基线42':
            return X
        extra = ZS if name == 'B +zs_pos' else RS
        return np.concatenate([X, extra[:, :, None]], axis=2)

    names = ['A 基线42', 'B +zs_pos', 'C +zs残差']
    rows = []
    for k in range(5):
        te0, te1 = bounds[k + 1], bounds[k + 2]
        tr0, tr1 = bounds[0], bounds[k + 1] - PURGE
        if k > 0:
            assert tr1 <= bounds[k + 1] - EMBARGO, 'embargo 冲突'
        mtr = np.zeros(T, bool); mtr[tr0:tr1:STEP] = True
        mte = np.zeros(T, bool); mte[te0:te1:STEP] = True
        tt, ii = np.where(M & mtr[:, None] & np.isfinite(Y))
        ta, ia = np.where(M & mte[:, None] & np.isfinite(Y))
        ytr = yrank(Y, M, tt, ii)
        rec = dict(fold=k + 1, val_from=str(dates[te0].date()), val_to=str(dates[te1 - 1].date()),
                   n_train=len(tt), n_val=len(ta))
        t0 = time.time()
        for tag in names:
            XA = arm(tag)
            ml = lgb.train(params, lgb.Dataset(XA[tt, ii, :], ytr), num_boost_round=NROUND)
            p = ml.predict(XA[ta, ia, :], num_iteration=NROUND)
            s = daily_ic(p, ta, ia, Y)
            rec[f'{tag}_ic'] = s.mean()
            rec[f'{tag}_icir'] = s.mean() / s.std()
            if tag != 'A 基线42':
                imp = ml.feature_importance('gain')
                rec[f'{tag}_imp_rank'] = int((imp > imp[-1]).sum()) + 1   # 新因子按 gain 排第几
            del XA, ml
        rec['secs'] = round(time.time() - t0, 1)
        rows.append(rec)
        print('折%d 验证 %s~%s | A %+.4f | B %+.4f（Δ%+.4f）| C %+.4f（Δ%+.4f）| zs增益排名 B#%s C#%s | %.0fs' % (
            k + 1, rec['val_from'], rec['val_to'],
            rec['A 基线42_ic'], rec['B +zs_pos_ic'], rec['B +zs_pos_ic'] - rec['A 基线42_ic'],
            rec['C +zs残差_ic'], rec['C +zs残差_ic'] - rec['A 基线42_ic'],
            rec['B +zs_pos_imp_rank'], rec['C +zs残差_imp_rank'], rec['secs']), flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{ROOT}/reports/chan_cv_folds.csv', index=False)

    print('\n=== 汇总（5 折均值 ± 标准差）===')
    for tag in names:
        s = d[f'{tag}_ic']
        print('%-10s %+.4f ± %.4f | 逐折 %s | 正折 %d/5' %
              (tag, s.mean(), s.std(), [round(v, 4) for v in s], int((s > 0).sum())), flush=True)
    for tag in ['B +zs_pos', 'C +zs残差']:
        diff = d[f'{tag}_ic'] - d['A 基线42_ic']
        t_ = sps.ttest_1samp(diff, 0)
        print('%s − A：逐折 Δ %s | 均值 %+.4f | t=%.2f | 下降折 %d/5' %
              (tag, [round(v, 4) for v in diff], diff.mean(), t_.statistic, int((diff < 0).sum())), flush=True)

    # ---- 报告 ----
    L = ['# 缠论 zs_pos：5 折 Purged CV 增量检验', '',
         '> %s｜协议与 `purged_cv.py` 完全一致（expanding 6 块、purge/embargo 6 日、STEP=5、'
         '固定 NROUND=85 不逐折选参）｜三臂：A=42 因子基线，B=+`chan_zs_pos` 原始值，C=+残差值'
         '（逐日截面剔除 momq60/ret60，不用标签）｜`zs_pos` 这次算的是**全市场 5,342 只**（之前 IC 用的 1,500 只抽样）' %
         pd.Timestamp.today().strftime('%Y-%m-%d'), '',
         '## 逐折结果', '',
         '| 折 | 验证段 | A 基线 IC | B +原始 IC | ΔB | C +残差 IC | ΔC | zs增益排名(B/C) |',
         '|---|---|---|---|---|---|---|---|']
    for _, r in d.iterrows():
        L.append('| %d | %s~%s | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | #%s / #%s |' % (
            r['fold'], r['val_from'], r['val_to'],
            r['A 基线42_ic'], r['B +zs_pos_ic'], r['B +zs_pos_ic'] - r['A 基线42_ic'],
            r['C +zs残差_ic'], r['C +zs残差_ic'] - r['A 基线42_ic'],
            r['B +zs_pos_imp_rank'], r['C +zs残差_imp_rank']))
    L += ['', '## 汇总', '', '| 臂 | 5 折 IC 均值 | 标准差 | 正折 |', '|---|---|---|---|']
    for tag in names:
        s = d[f'{tag}_ic']
        L.append('| %s | %+.4f | %.4f | %d/5 |' % (tag, s.mean(), s.std(), int((s > 0).sum())))
    for tag in ['B +zs_pos', 'C +zs残差']:
        diff = d[f'{tag}_ic'] - d['A 基线42_ic']
        t_ = sps.ttest_1samp(diff, 0)
        L.append('')
        L.append('**%s vs 基线**：逐折 Δ = %s，均值 **%+.4f**（t=%.2f，配对），下降折 **%d/5**。' % (
            tag, ['%+.4f' % v for v in diff], diff.mean(), t_.statistic, int((diff < 0).sum())))
    L += ['', '## 判定', '',
          '- 按项目规矩（≥4/5 折不下降且均值上升）：见上表数字，过/不过一目了然；',
          '- 若不过：`zs_pos` 留在「有单因子增量、无模型增量」一档，与 WQ101 价量背离同待遇——'
          '报告结论不变，但不进 42 因子面板；',
          '- 若过：仍需重跑 `big_train.py` 全管线 + 禁买清单评估，才能算正式入库。', '']
    open(f'{ROOT}/reports/chan_cv.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
    print('\n→', f'{ROOT}/reports/chan_cv.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='chan_cv'):
        main()
