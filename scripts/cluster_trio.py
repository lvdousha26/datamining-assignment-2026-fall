# -*- coding: utf-8 -*-
"""A3 · 聚类三件套：肘部法 / 轮廓系数 / bootstrap ARI

为什么做
--------
项目里一直在用「聚类 31 组」当行业中性化的分组（`ind_roll.parquet`），
但从没回答过三个问题：**k 取多少合适？分得开吗？换个样本还一样吗？**
本脚本用 42 个因子的截面画像对股票聚类，把这三件事各给一个数：

  1. 肘部法（inertia 随 k 的拐点）
  2. 轮廓系数（silhouette，k=2..12 全扫）
  3. bootstrap ARI（重抽样 100 次，看同一批股票还分不分成同一堆）

口径
----
- 画像：先做**逐日截面 z-score**（消掉全市场同涨同跌），再对每只股票在区间内取均值
  → 每股一个 42 维「性格向量」，与当期涨跌无关。
- 只用 2025H1（dev 段），不碰封存段标签；聚类本身是无监督的，但保持区间干净。
- 距离用欧氏，KMeans n_init=10；轮廓系数在 4000 只上算（O(n²) 可控）。

用法：python scripts/cluster_trio.py
产出：reports/cluster_trio.md + reports/cluster_trio.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
START, END = '2025-01-01', '2025-06-30'
KS = list(range(2, 13))
N_BOOT = 100
RNG = np.random.default_rng(0)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def profile_matrix(X, M, dates, start, end):
    """逐日截面 z-score → 区间均值 → 每股 42 维画像"""
    idx = np.where((dates >= start) & (dates <= end))[0]
    acc, cnt = None, None
    for t in idx:
        rows = np.where(M[t])[0]
        if len(rows) < 500:
            continue
        blk = X[t, rows, :]
        mu = np.nanmean(blk, axis=0)
        sd = np.nanstd(blk, axis=0) + 1e-9
        z = (blk - mu) / sd
        if acc is None:
            acc = np.zeros((X.shape[1], X.shape[2]), np.float64)
            cnt = np.zeros(X.shape[1], np.int32)
        acc[rows] += np.nan_to_num(z)
        cnt[rows] += 1
    ok = cnt >= 50
    prof = np.full_like(acc, np.nan)
    prof[ok] = acc[ok] / cnt[ok, None]
    return prof, ok, len(idx)


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, M = z['X'], z['M']
    dates = pd.to_datetime(z['dates'])
    feats = list(z['feats'])
    syms = np.array([str(s) for s in z['symbols']])

    prof, ok, ndays = profile_matrix(X, M, dates, START, END)
    P = prof[ok]
    S = syms[ok]
    keep = np.isfinite(P).all(1)
    P, S = P[keep], S[keep]
    P = np.clip(P, -5, 5)
    P = (P - P.mean(0)) / (P.std(0) + 1e-9)      # 维度间量纲拉平
    print('聚类样本：%d 只 × %d 因子（%s ~ %s，%d 个交易日）' % (len(P), P.shape[1], START, END, ndays))

    # ---- 1. 肘部 + 2. 轮廓系数 ----
    inert, sil = {}, {}
    for k in KS:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(P)
        inert[k] = km.inertia_
        s = silhouette_score(P, km.labels_, sample_size=min(4000, len(P)), random_state=0)
        sil[k] = s
        print('  k=%2d  inertia %12.0f  silhouette %.4f' % (k, inert[k], s), flush=True)

    # 肘部：用「折线最大曲率」自动找拐点（二阶差分最大）
    ks = np.array(KS, float)
    iv = np.array([inert[k] for k in KS])
    d2 = np.diff(iv, 2)
    k_elbow = int(ks[1:-1][np.argmax(d2)]) if len(d2) else KS[0]
    k_sil = max(sil, key=sil.get)

    # ---- 3. bootstrap ARI（在三个 k 上各做一遍：轮廓最优 / 肘部拐点 / 项目实际用的 31）----
    K_BOOT = sorted({k_sil, k_elbow, 31})
    ari_by_k, base_by_k = {}, {}
    for k in K_BOOT:
        base = KMeans(n_clusters=k, n_init=10, random_state=0).fit(P)
        base_by_k[k] = base
        aris = []
        for b in range(N_BOOT):
            sub = RNG.choice(len(P), int(len(P) * 0.8), replace=False)
            km = KMeans(n_clusters=k, n_init=10, random_state=b).fit(P[sub])
            aris.append(adjusted_rand_score(base.labels_[sub], km.labels_))
        ari_by_k[k] = np.array(aris)
        print('  k=%2d  bootstrap ARI 均值 %.3f | 5%% 分位 %.3f | 最小 %.3f' % (
            k, ari_by_k[k].mean(), np.percentile(ari_by_k[k], 5), ari_by_k[k].min()), flush=True)

    sil31 = silhouette_score(P, base_by_k[31].labels_, sample_size=min(4000, len(P)), random_state=0)
    aris = ari_by_k[k_sil]
    print('\n最佳轮廓 k=%d（%.4f）| 肘部拐点 k=%d | k=31 轮廓 %.4f' % (k_sil, sil[k_sil], k_elbow, sil31))
    for k in K_BOOT:
        print('  k=%2d bootstrap ARI：均值 %.3f | 中位 %.3f | 5%% 分位 %.3f' % (
            k, ari_by_k[k].mean(), np.median(ari_by_k[k]), np.percentile(ari_by_k[k], 5)))

    # ---- 图 ----
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax[0].plot(KS, [inert[k] for k in KS], 'o-', color='#c0392b')
    ax[0].axvline(k_elbow, ls='--', c='gray', lw=1)
    ax[0].set_title('肘部法：inertia vs k\n拐点 k=%d' % k_elbow)
    ax[0].set_xlabel('k'); ax[0].set_ylabel('簇内平方和')
    ax[1].plot(KS, [sil[k] for k in KS], 'o-', color='#2980b9')
    ax[1].axvline(k_sil, ls='--', c='gray', lw=1)
    ax[1].set_title('轮廓系数 vs k\n最优 k=%d' % k_sil)
    ax[1].set_xlabel('k'); ax[1].set_ylabel('silhouette')
    ax[2].hist(aris, bins=20, color='#27ae60', alpha=.8)
    ax[2].axvline(aris.mean(), color='k', ls='--', lw=1)
    ax[2].set_title('bootstrap ARI @k=%d（%d 次）\n均值 %.3f' % (k_sil, N_BOOT, aris.mean()))
    ax[2].set_xlabel('ARI')
    for a in ax:
        a.grid(alpha=.2)
    plt.tight_layout()
    png = f'{ROOT}/reports/cluster_trio.png'
    plt.savefig(png, dpi=150)
    print('图 →', png)

    # ---- 报告 ----
    top_feats = None
    lab = base_by_k[k_sil].labels_
    cen = pd.DataFrame(P, columns=feats).groupby(lab).mean()
    # 每个簇最突出的 3 个因子（|z| 最大）
    desc = {}
    for c in cen.index:
        s = cen.loc[c].sort_values(key=np.abs, ascending=False).head(3)
        desc[c] = '、'.join('%s(%+.2f)' % (i, v) for i, v in s.items())
    sizes = pd.Series(lab).value_counts().sort_index()

    md = ['# A3 · 聚类三件套（肘部 / 轮廓 / bootstrap ARI）', '',
          '> %s｜样本 **%d 只 × %d 因子**｜区间 %s ~ %s（%d 个交易日，dev 段）' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), len(P), len(feats), START, END, ndays), '',
          '## 口径', '',
          '1. 每只股票先做**逐日截面 z-score**（消掉全市场同涨同跌），再对区间取均值 → 42 维「性格向量」，与当期涨跌无关；',
          '2. 维度间再标准化，欧氏距离，KMeans（n_init=10）；',
          '3. 轮廓系数在 4000 只上算（全量 O(n²) 没必要）；',
          '4. bootstrap：抽 80% 股票重聚类 100 次，与全样本标签算 ARI（1=完全一致，0=随机）。', '',
          '## 结果', '',
          '| 方法 | 结论 |', '|---|---|',
          '| 肘部法 | 拐点 **k=%d** |' % k_elbow,
          '| 轮廓系数 | 最优 **k=%d**（%.4f） |' % (k_sil, sil[k_sil]),
          '| bootstrap ARI @k=%d | 均值 **%.3f**，5%% 分位 %.3f |' % (k_sil, aris.mean(), np.percentile(aris, 5)), '',
          '| bootstrap ARI @k=31（项目实际用的组数） | 均值 **%.3f**，5%% 分位 %.3f |' % (
              ari_by_k[31].mean(), np.percentile(ari_by_k[31], 5)), '',
          '| k | 轮廓系数 | bootstrap ARI 均值 | 5% 分位 |', '|---|---|---|---|']
    for k in KS:
        md.append('| %d | %.4f | %s | %s |' % (
            k, sil[k],
            ('%.3f' % ari_by_k[k].mean()) if k in ari_by_k else '—',
            ('%.3f' % np.percentile(ari_by_k[k], 5)) if k in ari_by_k else '—'))
    md += ['',
          '| k | inertia | silhouette |', '|---|---|---|']
    for k in KS:
        md.append('| %d | %s | %.4f |' % (k, f'{inert[k]:,.0f}', sil[k]))
    md += ['', '## k=%d 的各簇画像（前 3 个最突出因子）' % k_sil, '',
           '| 簇 | 只数 | 占比 | 特征 |', '|---|---|---|---|']
    for c in cen.index:
        md.append('| %d | %d | %.1f%% | %s |' % (c, sizes[c], sizes[c] / len(P) * 100, desc[c]))
    md += ['', '## 怎么读这三个数', '',
           '- **ARI 才是关键**。轮廓系数只说明「这次分得开」，ARI 说明「换一批股票还分不分得开」。'
           'ARI < 0.6 意味着聚类结果对样本高度敏感，拿它做中性化分组要打折扣。',
           '- 本项目把聚类 31 组当行业代理，但那 31 组是按**收益相关性**聚的（`ind_rolling.py`），'
           '这里的 k 是按**因子画像**聚的，两个数不冲突，但**不能互相替代**：前者管风险，后者管风格。',
           '- 轮廓系数在 5000 只、42 维上天然偏低（0.1~0.3 是常态），不要拿它跟二维玩具数据比。', '']
    out = f'{ROOT}/reports/cluster_trio.md'
    open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('报告 →', out)


if __name__ == '__main__':
    with RunLogger(__file__, tag='cluster'):
        main()
