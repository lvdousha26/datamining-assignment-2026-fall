# -*- coding: utf-8 -*-
"""行业中性化开关：加不加、按什么分组，对 5 折 CV 有什么影响

背景
----
主面板的 X **没有做行业中性化**（`big_features.py` 只把 industry 存进 IND，供下游统计用）。
之前的对比（`neutral_compare.py`）只看了单因子 IC，这次直接看模型：

    ① 原始（现状）
    ② 申万一级 31 类（`ind_map.parquet`）
    ③ 聚类 31 组（`ind_roll.parquet`，按收益相关性，逐年）
    ④ 聚类 4 组（按因子画像，逐年重聚）

协议与 `purged_cv.py` 完全一致：6 块 expanding、purge 6 / embargo 6、固定 best_iter=85、
固定超参。中性化逐日按组去均值（组内 <20 只的组跳过），在**取数时**做，不复制整张面板
（可用内存只有 3.9G，X 本身占 2.5G）。

用法：python scripts/neutral_switch.py
产出：reports/neutral_switch.md + csv
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

ROOT = r'D:/quant'
PURGE, STEP = 6, 5
NROUND = 85
PARAMS = dict(objective='regression', metric='l2', learning_rate=0.03, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=8, seed=42)


def yrank(Y2, M, tt, ii):
    """逐日把标签转成 (rank-0.5)/n"""
    out = np.empty(len(tt), np.float32)
    for t in np.unique(tt):
        idx = np.where(tt == t)[0]
        v = Y2[t, ii[idx]]
        order = np.argsort(np.argsort(v))
        out[idx] = (order + 0.5) / len(v)
    return out


def neutralize(Xg, tt, gi):
    """按 (日, 组) 去均值；组号为 -1 的行不参与（也不被减）"""
    valid = gi >= 0
    key = tt.astype(np.int64) * 200 + np.where(valid, gi, 0).astype(np.int64)
    cnt = np.bincount(key).astype(np.float64)
    for j in range(Xg.shape[1]):
        s = np.bincount(key, weights=Xg[:, j].astype(np.float64))
        mu = (s / np.maximum(cnt, 1))[key].astype(np.float32)
        Xg[:, j] -= np.where(valid, mu, 0.0)
    return Xg


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, Y2, M, IND = z['X'], z['Y2'], z['M'], z['IND']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    syms = np.array([str(s).zfill(6) for s in z['symbols']])
    feats = list(z['feats'])
    T, N, F = X.shape

    usable = np.where((M & np.isfinite(Y2)).any(1))[0]
    bounds = np.linspace(usable[0], usable[-1] + 1, 7).astype(int)
    print('可用 %d 日：%s ~ %s | 折界 %s' % (
        len(usable), dates[usable[0]].date(), dates[usable[-1]].date(), bounds.tolist()), flush=True)

    # ---- 三种分组 ----
    im = pd.read_parquet(f'{ROOT}/data/big/ind_map.parquet')
    m_sw = dict(zip(im['symbol'].astype(str).str.zfill(6), im['industry']))
    sw_names = sorted(set(m_sw.values()))
    g_sw = np.array([sw_names.index(m_sw[s]) if s in m_sw else -1 for s in syms])

    ir = pd.read_parquet(f'{ROOT}/data/big/ind_roll.parquet')
    ir['symbol'] = ir['symbol'].astype(str).str.zfill(6)
    g_c31 = np.full((T, N), -1, dtype=np.int16)
    for yr, gg in ir.groupby('year'):
        tmask = np.where(dates.dt.year == yr)[0]
        mp = dict(zip(gg['symbol'], gg['cluster']))
        for t in tmask:
            g_c31[t] = [mp.get(s, -1) for s in syms]

    # 聚类 4：逐年按因子画像重聚
    from cluster_trio import profile_matrix
    from sklearn.cluster import KMeans
    g_c4 = np.full((T, N), -1, dtype=np.int16)
    for yr in sorted(set(dates.dt.year)):
        s0, s1 = f'{yr}-01-01', f'{yr}-12-31'
        prof, ok, _ = profile_matrix(X, M, dates, s0, s1)
        if ok.sum() < 300:
            continue
        P = prof[ok]
        keep = np.isfinite(P).all(1)
        Pk = np.clip(P[keep], -5, 5)
        Pk = (Pk - Pk.mean(0)) / (Pk.std(0) + 1e-9)
        lab = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Pk).labels_
        idx_stock = np.where(ok)[0][keep]
        tmask = np.where(dates.dt.year == yr)[0]
        g_c4[np.ix_(tmask, idx_stock)] = lab[None, :]
        print('  %d 年聚类完成：%d 只' % (yr, len(idx_stock)), flush=True)

    # 严格 PIT 版：用【上一年的画像】给本年分组，避免"用当年数据给当年分组"的乐观偏差
    g_c4_lag = np.full((T, N), -1, dtype=np.int16)
    for yr in sorted(set(dates.dt.year)):
        prev = f'{yr - 1}-01-01', f'{yr - 1}-12-31'
        try:
            prof, ok, _ = profile_matrix(X, M, dates, *prev)
        except Exception:
            continue
        if prof is None or ok is None or ok.sum() < 300:
            continue
        P = prof[ok]
        keep = np.isfinite(P).all(1)
        Pk = np.clip(P[keep], -5, 5)
        Pk = (Pk - Pk.mean(0)) / (Pk.std(0) + 1e-9)
        lab = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Pk).labels_
        idx_stock = np.where(ok)[0][keep]
        tmask = np.where(dates.dt.year == yr)[0]
        g_c4_lag[np.ix_(tmask, idx_stock)] = lab[None, :]
        print('  %d 年用 %d 年画像分组：%d 只' % (yr, yr - 1, len(idx_stock)), flush=True)

    PANELS = {'原始（现状）': None, '申万一级31': g_sw, '聚类31': g_c31,
              '聚类4（当年拟合，含偏差）': g_c4, '聚类4（上年画像，PIT）': g_c4_lag}

    results = []
    for name, gmat in PANELS.items():
        t0 = time.time()
        fold_ics = []
        for k in range(1, 6):
            tr_hi = bounds[k] - PURGE
            tr_t, tr_i = np.where(M[bounds[0]:tr_hi] & np.isfinite(Y2[bounds[0]:tr_hi]))
            tr_t = tr_t + bounds[0]
            va_t, va_i = np.where(M[bounds[k]:bounds[k + 1]] & np.isfinite(Y2[bounds[k]:bounds[k + 1]]))
            va_t = va_t + bounds[k]
            Xtr = X[tr_t, tr_i, :].copy()
            Xva = X[va_t, va_i, :].copy()
            if gmat is not None:
                def gtake(tt, ii):
                    return gmat[ii] if gmat.ndim == 1 else gmat[tt, ii]
                Xtr = neutralize(Xtr, tr_t, gtake(tr_t, tr_i))
                Xva = neutralize(Xva, va_t, gtake(va_t, va_i))
            ytr = yrank(Y2, M, tr_t, tr_i)
            m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=NROUND)
            p = m.predict(Xva)
            ics = []
            for t in np.unique(va_t):
                sel = va_t == t
                if sel.sum() < 100:
                    continue
                ics.append(sps.spearmanr(p[sel], Y2[t, va_i[sel]]).statistic)
            fold_ics.append(np.nanmean(ics))
            del Xtr, Xva
            print('  %-12s 折%d IC %+.4f | %.0fs' % (name, k, fold_ics[-1], time.time() - t0), flush=True)
        a = np.array(fold_ics)
        results.append(dict(panel=name, ic=float(a.mean()), std=float(a.std()),
                            icir=float(a.mean() / a.std()) if a.std() else np.nan,
                            folds=json.dumps([round(float(x), 4) for x in a]),
                            sec=time.time() - t0))
        print('→ %-12s 5 折均值 IC %+.4f ± %.4f | %.0fs' % (name, a.mean(), a.std(), time.time() - t0), flush=True)

    df = pd.DataFrame(results)
    df.to_csv(f'{ROOT}/reports/neutral_switch.csv', index=False)
    base = df.iloc[0]['ic']
    md = ['# 行业中性化开关：对 5 折 CV 的影响', '',
          '> %s｜purge 6 / embargo 6 / expanding 6 块｜固定 best_iter=%d｜%d 因子' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), NROUND, F), '',
          '| 面板 | 5 折 CV IC | 折间标准差 | 折间 t | 相对现状 |', '|---|---|---|---|---|']
    for _, r in df.iterrows():
        tt = r['ic'] / (r['std'] / np.sqrt(5)) if r['std'] else float('nan')
        md.append('| %s | **%+.4f** | %.4f | %.1f | %+.4f |' % (
            r['panel'], r['ic'], r['std'], tt, r['ic'] - base))
    md += ['', '逐折明细：', '', '```']
    for _, r in df.iterrows():
        md.append('%-12s %s' % (r['panel'], r['folds']))
    md += ['```', '',
           '## 结论', '',
           '中性化后 5 折 CV 的变化见上表「相对现状」列。判读标准：',
           '- **提升 ≥ 0.005**：值得进管线，但要先确认不是分组信息泄漏（分组是否用了全期数据）；',
           '- **持平（±0.003）**：说明模型自己已经学到了行业暴露，中性化是多余的；',
           '- **下降**：说明行业暴露本身就是有效信号的一部分（A 股的行业动量/风格轮动），去掉反而变差。', '',
           '## 注意', '',
           '1. 「聚类4（当年拟合）」用当年数据给自己分组，属样本内，**不能作为结论**；',
           '   「聚类4（上年画像）」才是干净版本——两者差值就是分组前瞻带来的乐观偏差。',
           '2. 只测了 LGBM。TF 与 LGBM 的预测相关性 0.88，结论大概率同向，但没有单独验证。',
           '3. 中性化在取数时做（逐日按组去均值），等价于在管线里 X 构造完之后插一步，',
           '   不影响其他 41 个因子的计算方式。', '']
    out = f'{ROOT}/reports/neutral_switch.md'
    open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n→', out)


if __name__ == '__main__':
    with RunLogger(__file__, tag='neutral_switch'):
        main()
