# -*- coding: utf-8 -*-
"""A4 · SHAP 解释：全局重要性 + 3 个样本的局部归因

为什么不用 `pip install shap`
----------------------------
LightGBM 自带 `pred_contrib=True`，输出的就是**精确 TreeSHAP 值**（不是近似），
不需要额外依赖。这个项目刚被 `pip install laya` 顺带升级 torch 坑过一次，
能不加依赖就不加。

口径
----
- 模型：`checkpoints/lgbm_big.txt`（best_iter 冻结），只用它解释，不重训；
- 样本：2025H1（dev 段）随机 2 万行，全部有标签区间，不碰封存段；
- SHAP 值满足 `base_value + Σ contrib = 预测值`，代码里会断言误差 < 1e-4。

用法：python scripts/shap_explain.py
产出：reports/shap_explain.md + reports/shap_global.png
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
START, END = '2025-01-01', '2025-06-30'
N = 20000
RNG = np.random.default_rng(7)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, M = z['X'], z['M']
    dates = pd.to_datetime(z['dates'])
    feats = list(z['feats'])
    syms = np.array([str(s) for s in z['symbols']])

    lmeta = json.load(open(f'{ROOT}/checkpoints/lgbm_meta.json'))
    booster = lgb.Booster(model_file=f'{ROOT}/checkpoints/lgbm_big.txt')
    best_iter = lmeta.get('best_iter', booster.best_iteration or booster.num_trees())

    d0 = np.where((dates >= START) & (dates <= END))[0][0]
    d1 = np.where((dates >= START) & (dates <= END))[0][-1]
    tt, ii = np.where(M[d0:d1 + 1])
    tt = tt + d0
    sel = RNG.choice(len(tt), min(N, len(tt)), replace=False)
    tt, ii = tt[sel], ii[sel]
    Xs = X[tt, ii, :]
    print('样本 %d 行 × %d 因子（%s ~ %s）| best_iter=%d' % (
        len(Xs), Xs.shape[1], dates[d0].date(), dates[d1].date(), best_iter), flush=True)

    contrib = booster.predict(Xs, pred_contrib=True, num_iteration=best_iter)
    base = contrib[:, -1]
    phi = contrib[:, :-1]
    pred = booster.predict(Xs, num_iteration=best_iter)
    err = np.abs(base + phi.sum(1) - pred).max()
    print('SHAP 可加性校验：max|base+Σφ−pred| = %.2e' % err, flush=True)
    assert err < 1e-4, 'SHAP 值不满足可加性，说明 predict 参数与模型不一致'

    imp = np.abs(phi).mean(0)
    order = np.argsort(-imp)
    norm = imp / imp.sum()
    print('\n全局重要性 Top 10：')
    for r, j in enumerate(order[:10], 1):
        print('  %2d. %-12s mean|SHAP| %.5f（占 %.1f%%）' % (r, feats[j], imp[j], norm[j] * 100))

    # ---- 与线性口径对比 ----
    meta = json.load(open(f'{ROOT}/reports/big_models_meta.json'))
    w = np.array([meta['weights'].get(f, 0.0) for f in feats])
    ic = np.array([meta['factor_ic'].get(f, np.nan) if isinstance(meta.get('factor_ic'), dict) else np.nan
                   for f in feats])
    rho_ic = spearmanr(imp[np.isfinite(ic)], np.abs(ic[np.isfinite(ic)])).statistic
    rank = {feats[j]: r for r, j in enumerate(order, 1)}
    nz = [(feats[j], w[j]) for j in np.argsort(-w) if w[j] > 0]
    print('\nSHAP 重要性 vs |单因子 IC|：Spearman %.3f' % rho_ic)
    print('线性合成权重非零的 %d 个因子，在 SHAP 里的排名：' % len(nz))
    for f, ww in nz:
        print('   %-12s 权重 %.1f%% → SHAP 第 %2d 名（占 %.1f%%）' % (f, ww * 100, rank[f], norm[feats.index(f)] * 100))

    # ---- 三个样本 ----
    pick = {'最高分': int(np.argmax(pred)), '中位': int(np.argsort(pred)[len(pred) // 2]), '最低分': int(np.argmin(pred))}
    rows_md = []
    for name, i in pick.items():
        pct = np.argsort(np.argsort(phi[i]))[::-1]  # 不看，只用于排序展示
        top = np.argsort(-np.abs(phi[i]))[:6]
        # 当日同因子的截面分位（帮助读懂"高/低"）
        day = tt[i]
        peer = X[day, M[day], :]
        pctile = [(peer[:, j] < Xs[i, j]).mean() for j in top]
        lines = ['| 因子 | 取值 | 截面分位 | SHAP 贡献 |', '|---|---|---|---|']
        for j, p in zip(top, pctile):
            lines.append('| %s | %.3f | %.0f%% | %+.4f |' % (feats[j], Xs[i, j], p * 100, phi[i, j]))
        rows_md.append((name, syms[ii[i]], dates[tt[i]].date(), pred[i], base[i], lines))
        print('\n[%s] %s %s 预测 %.4f（基准 %.4f）' % (name, syms[ii[i]], dates[tt[i]].date(), pred[i], base[i]))
        for j, p in zip(top, pctile):
            print('   %-12s 值 %8.3f 分位 %3.0f%%  φ %+.4f' % (feats[j], Xs[i, j], p * 100, phi[i, j]))

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(7.2, 6))
    top = order[:15][::-1]
    ax.barh([feats[j] for j in top], imp[top], color='#2c7fb8')
    ax.set_xlabel('mean |SHAP|（对预测值的平均绝对贡献）')
    ax.set_title('全局重要性 Top 15（LGBM，2025H1 2 万行）')
    ax.grid(axis='x', alpha=.2)
    plt.tight_layout()
    png = f'{ROOT}/reports/shap_global.png'
    plt.savefig(png, dpi=150)

    # ---- 报告 ----
    md = ['# A4 · SHAP 解释（LGBM，TreeSHAP 精确值）', '',
          '> %s｜样本 **%d 行**（2025H1 dev 段）｜模型 `lgbm_big.txt`（best_iter=%d）冻结，未重训' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), len(Xs), best_iter), '',
          '## 0. 方法', '',
          '- 用 LightGBM 自带 `pred_contrib=True` 得到**精确** TreeSHAP，不引入 `shap` 依赖；',
          '- 校验：`max|base + Σφ − pred| = %.1e`（< 1e-4 才算对）' % err, '',
          '## 1. 全局重要性 Top 15', '',
          '| # | 因子 | mean\\|SHAP\\| | 占比 |', '|---|---|---|---|']
    for r, j in enumerate(order[:15], 1):
        md.append('| %d | `%s` | %.5f | %.1f%% |' % (r, feats[j], imp[j], norm[j] * 100))
    md += ['', '## 2. 三个样本的局部归因', '']
    for name, sym, dt, p, b, lines in rows_md:
        md += ['### %s：%s（%s）' % (name, sym, dt), '',
               '预测 %.4f，基准值 %.4f；下表是贡献最大的 6 个因子（截面分位 = 当日该因子在 5000 只里的位置）。' % (p, b), '']
        md += lines + ['']
    md += ['## 3. SHAP 与线性口径的差别', '',
           '- SHAP 重要性 vs |单因子 IC|：**Spearman %.3f**（弱正相关）' % rho_ic, '',
           '线性合成（ICIR 加权）只给 6 个因子非零权重，其余 36 个为 0；它们在 SHAP 里的位置是：', '',
           '| 因子 | 线性权重 | SHAP 排名 | SHAP 占比 |', '|---|---|---|---|']
    for f, ww in nz:
        md.append('| `%s` | %.1f%% | 第 %d 名 | %.1f%% |' % (f, ww * 100, rank[f], norm[feats.index(f)] * 100))
    md += ['',
           '**读法**：线性口径把 67%% 的权重压在 `illiq_ratio` + `illiq20` 上，'
           '但树模型的 SHAP 前三是 `size`(%.1f%%)、`turn5`(%.1f%%)、`pos250`(%.1f%%)，'
           '`illiq_ratio` 只排第 %d。这不是矛盾——四个因子是同一条「流动性」主轴的四种打法：'
           '线性要一个单调排序，树模型要若干阈值切分。**结论仍是 42 个因子塌缩成流动性**，'
           '只是不同模型抓的是这条轴的不同切面。' % (
               norm[order[0]] * 100, norm[order[1]] * 100, norm[order[2]] * 100,
               rank.get('illiq_ratio', -1)), '',
           'SHAP 与 |IC| 只有 %.2f 的秩相关，说明「单因子有效」和「模型在用」是两件事：'
           'IC 高的因子未必被模型用，被模型用的因子未必 IC 高（它可能只在交互里起作用）。' % rho_ic, '',
           '## 4. 局限', '',
           '1. SHAP 解释的是**模型**，不是**市场**：它只说"模型为什么这么打分"，不说"股价为什么涨"；',
           '2. 特征之间高度相关（42 个因子塌缩成流动性），SHAP 会把贡献在共线因子间摊分，'
           '单看某一个因子的 SHAP 会被低估；',
           '3. 样本只取 2025H1，换一段时期排名会变（尤其 `illiq` 这类状态型因子）。', '']
    out = f'{ROOT}/reports/shap_explain.md'
    open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n报告 →', out)


if __name__ == '__main__':
    with RunLogger(__file__, tag='shap'):
        main()
