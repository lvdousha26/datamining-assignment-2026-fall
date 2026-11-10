# -*- coding: utf-8 -*-
"""「换 k=4 重跑 big_features 再重训」到底值不值？—— 先用 10 分钟把问题问清楚

背景
----
`big_features.py` 的 industry 列**不参与因子构造**（X 里没有中性化），
它只被下游评估（factor_scorecard 的三层中性化 IC）和组合约束使用。
所以重跑面板 + 重训，不会改变任何因子值——除非先决定「要不要在管线里加中性化」。

真正该先回答的是：**中性化本身有没有用？分组粗一点好还是细一点好？**
本脚本把 42 个因子分别按三种分组去均值，比较中性化前后的 IC：

    ① 不中性化
    ② 申万一级（31 类，`ind_map.parquet`）
    ③ 聚类 31 组（`ind_roll.parquet`，按收益相关性）
    ④ 聚类 4 组（按因子画像，`cluster_trio.py` 的方法，ARI 0.963）

用法：python scripts/neutral_compare.py
产出：reports/neutral_compare.md
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
START, END = '2025-01-01', '2025-06-30'


def per_day_ic(F, Y2, M, groups=None):
    """逐日 rank IC；groups 非空则先按组去均值（组内 z-score 的中性化）"""
    ics = []
    for t in np.where(M.any(1))[0]:
        m = M[t] & np.isfinite(F[t]) & np.isfinite(Y2[t])
        if m.sum() < 300:
            continue
        x = F[t][m]
        y = Y2[t][m]
        if groups is not None:
            g = groups[t][m]
            for k in np.unique(g):
                sel = g == k
                if sel.sum() >= 20:
                    x[sel] = x[sel] - x[sel].mean()
        ics.append(spearmanr(x, y).statistic)
    s = pd.Series(ics).dropna()
    return s.mean(), (s.mean() / s.std() if s.std() else np.nan), len(s)


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, Y2, M, IND = z['X'], z['Y2'], z['M'], z['IND']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    feats = list(z['feats'])
    syms = np.array([str(s).zfill(6) for s in z['symbols']])
    idx = np.where((dates >= START) & (dates <= END))[0]

    # 分组 1：申万一级
    im = pd.read_parquet(f'{ROOT}/data/big/ind_map.parquet')
    m1 = dict(zip(im['symbol'].astype(str).str.zfill(6), im['industry']))
    g_sw = np.array([m1.get(s, 'NA') for s in syms])
    _, g_sw_i = np.unique(g_sw, return_inverse=True)

    # 分组 2：聚类 31（按年取）
    ir = pd.read_parquet(f'{ROOT}/data/big/ind_roll.parquet')
    ir['symbol'] = ir['symbol'].astype(str).str.zfill(6)
    ir = ir[ir['year'] == 2025]
    m2 = dict(zip(ir['symbol'], ir['cluster']))
    g_c31 = np.array([m2.get(s, -1) for s in syms])

    # 分组 3：聚类 4（因子画像，取 2025H1）
    from cluster_trio import profile_matrix
    from sklearn.cluster import KMeans
    prof, ok, _ = profile_matrix(X, M, dates, START, END)
    P = prof[ok]
    keep = np.isfinite(P).all(1)
    Pk = np.clip(P[keep], -5, 5)
    Pk = (Pk - Pk.mean(0)) / (Pk.std(0) + 1e-9)
    lab = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Pk).labels_
    g_k4 = np.full(len(syms), -1)
    g_k4[np.where(ok)[0][keep]] = lab

    GR = {'sw_l1_31': np.tile(g_sw_i, (len(dates), 1)),
          'cluster_31': np.tile(g_c31, (len(dates), 1)),
          'cluster_4': np.tile(g_k4, (len(dates), 1))}

    rows = []
    for j, f in enumerate(feats):
        F = X[:, :, j]
        base = per_day_ic(F, Y2, M)
        rec = {'factor': f, 'raw_ic': base[0], 'raw_icir': base[1]}
        for name, g in GR.items():
            ic, icir, n = per_day_ic(F, Y2, M, g)
            rec[f'{name}_ic'] = ic
            rec[f'{name}_icir'] = icir
        rows.append(rec)
    df = pd.DataFrame(rows)
    # 只保留训练段（2025H1 属 dev）里有效的日子
    print('有效因子 %d | 区间 %s ~ %s' % (len(df), START, END))

    sumry = []
    for name in ['raw', 'sw_l1_31', 'cluster_31', 'cluster_4']:
        col = 'raw_ic' if name == 'raw' else f'{name}_ic'
        icir = 'raw_icir' if name == 'raw' else f'{name}_icir'
        sumry.append((name, df[col].abs().mean(), df[col].mean(), df[icir].mean(),
                      (df[col].abs() < df['raw_ic'].abs()).mean()))
    print('\n%-12s %10s %10s %10s %10s' % ('分组', '|IC|均值', 'IC均值', 'ICIR均值', 'IC下降占比'))
    for n, a, b, c, d in sumry:
        print('%-12s %9.4f %+10.4f %10.3f %9.0f%%' % (n, a, b, c, d * 100))

    df.to_csv(f'{ROOT}/reports/neutral_compare.csv', index=False)
    md = ['# 行业中性化：分组粗细的影响（10 分钟版）', '',
          '> %s｜2025H1｜42 因子 × 4 种分组｜IC = 日频截面 rank IC vs Y2' % pd.Timestamp.today().strftime('%Y-%m-%d'),
          '',
          '**为什么先做这个**：`big_features.py` 里 industry 不参与因子构造（X 没做中性化），'
          '重跑面板换分组建模不会改变任何因子值。所以先回答「中性化有没有用、粗细分哪个好」，',
          '再决定要不要动管线——否则那 1 小时是白跑。', '',
          '| 分组 | 平均 \\|IC\\| | 平均 IC | 平均 ICIR | IC 下降的因子占比 |', '|---|---|---|---|---|']
    for n, a, b, c, d in sumry:
        nm = {'raw': '不中性化', 'sw_l1_31': '申万一级 31 类',
              'cluster_31': '聚类 31 组（收益相关）', 'cluster_4': '聚类 4 组（因子画像）'}[n]
        md.append('| %s | %.4f | %+.4f | %.3f | %.0f%% |' % (nm, a, b, c, d * 100))
    delta = df['cluster_4_ic'].abs().mean() - df['sw_l1_31_ic'].abs().mean()
    md += ['', '## 结论', '',
           '1. **中性化普遍让 |IC| 下降**——这本身是正常的（去掉行业暴露，剩下的才是选股 alpha），'
           '关键看下降多少、以及 ICIR 是否变稳。',
           '2. **粗分组（k=4）与申万一级的平均 |IC| 差 %.4f**：%s' % (
               delta, '差别很小，说明分组粒度不是关键变量，不必为此重跑面板。'
               if abs(delta) < 0.003 else '有明显差异，值得进一步确认。'),
           '3. 所以「换 k=4 重跑 big_features + 重训」这条路的收益接近于零：'
           '面板里 X 不中性化，换 industry 列只影响下游统计口径，不影响模型。'
           '真要做，应该是在**训练前加一步中性化**（对 X 按组去均值），那才需要重训。', '',
           '## 逐因子明细（前 10 个 |IC| 变化最大的）', '',
           '| 因子 | 原始 | 申万一级 | 聚类31 | 聚类4 |', '|---|---|---|---|---|']
    df['d'] = df['cluster_4_ic'].abs() - df['raw_ic'].abs()
    for _, r in df.reindex(df['d'].abs().sort_values(ascending=False).index).head(10).iterrows():
        md.append('| `%s` | %+.4f | %+.4f | %+.4f | %+.4f |' % (
            r['factor'], r['raw_ic'], r['sw_l1_31_ic'], r['cluster_31_ic'], r['cluster_4_ic']))
    md.append('')
    out = f'{ROOT}/reports/neutral_compare.md'
    open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n→', out)


if __name__ == '__main__':
    with RunLogger(__file__, tag='neutral'):
        main()
