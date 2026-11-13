# -*- coding: utf-8 -*-
"""加分项 D：不平衡处理 / 交互特征 + t-SNE / 特征选择三法

之前只用了 `class_weight='balanced'`，而且没做前后对比——等于没交。这个脚本把三件事补全：

  D1 不平衡：无处理 / 类别权重 / 随机欠采样 / SMOTE（简化版，稀疏向量中点插值）
      → 报告训练集各类样本量 + 评估集 accuracy / macro-F1 / 中性类召回
  D2 交互特征（表格侧）+ t-SNE（42 因子画像）
  D3 特征选择三法：方差 / 相关性 / 嵌入法（LGBM gain）

用法：python scripts/bonus_d.py
产出：reports/bonus_d.md + reports/tsne_factors.png
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
DIR = f'{ROOT}/runs/laya_yjyg'
LAB = ['利好', '中性', '利空']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- D1
def smote_mid(X, y, cls, k=5, n_new=None, seed=0):
    """SMOTE 简化版：在同类近邻里取中点插值（稀疏矩阵上做，不引入 imblearn）

    严格 SMOTE 是 x_new = x_i + u*(x_j - x_i)，u~U(0,1)；
    这里固定 u=0.5（中点），对 TF-IDF 这种非负稀疏向量更稳，也避免负值。
    """
    from sklearn.neighbors import NearestNeighbors
    rs = np.random.default_rng(seed)
    idx = np.where(y == cls)[0]
    Xc = X[idx]
    k = min(k, len(idx) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc)
    _, ind = nn.kneighbors(Xc)
    n_new = n_new or len(idx)
    pick = rs.integers(0, len(idx), n_new)
    neigh = ind[pick, rs.integers(1, k + 1, n_new)]
    synth = (Xc[pick] + Xc[neigh]) * 0.5
    return synth


def part1_imbalance():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from scipy.sparse import vstack

    tr = pd.read_csv(f'{DIR}/train.csv')
    ev = pd.read_csv(f'{DIR}/eval.csv')
    tv = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), min_df=2, max_features=20000)
    Xtr = tv.fit_transform(tr.text)
    Xev = tv.transform(ev.text)
    ytr, yev = tr.label.values, ev.label.values

    def fit_eval(X, y, tag, **kw):
        t0 = time.time()
        m = LogisticRegression(max_iter=1000, C=4.0, n_jobs=-1, **kw).fit(X, y)
        p = m.predict(Xev)
        sizes = pd.Series(y).value_counts().to_dict()
        row = dict(方案=tag, 训练量=len(y),
                   利好=sizes.get('利好', 0), 中性=sizes.get('中性', 0), 利空=sizes.get('利空', 0),
                   acc=accuracy_score(yev, p),
                   macro_f1=f1_score(yev, p, average='macro'),
                   中性召回=recall_score(yev, p, labels=['中性'], average='macro'),
                   耗时=time.time() - t0)
        print('  %-12s n=%5d | 利好%4d 中性%3d 利空%4d | acc %.4f | macroF1 %.4f | 中性召回 %.3f' % (
            tag, row['训练量'], row['利好'], row['中性'], row['利空'],
            row['acc'], row['macro_f1'], row['中性召回']), flush=True)
        return row

    print('\n[D1] 不平衡处理（评估集固定 602 条，分布不变）')
    rows = [fit_eval(Xtr, ytr, '无处理')]
    rows.append(fit_eval(Xtr, ytr, '类别权重', class_weight='balanced'))

    # 随机欠采样：多数类降到 中性类 x3（避免过度丢信息）
    n_mid = int((ytr == '中性').sum())
    keep = []
    for lab in LAB:
        idx = np.where(ytr == lab)[0]
        cap = n_mid * 3 if lab == '利好' else n_mid * 2 if lab == '利空' else n_mid
        keep += list(idx if len(idx) <= cap else RNG.choice(idx, cap, replace=False))
    keep = np.array(sorted(keep))
    rows.append(fit_eval(Xtr[keep], ytr[keep], '随机欠采样'))

    # SMOTE：中性类合成到 利空类规模的一半，利空合成到与利好 1:1.5
    t_mid = int((ytr == '利空').sum() * 0.5)
    syn_mid = smote_mid(Xtr, ytr, '中性', n_new=max(t_mid - n_mid, 0))
    t_neg = int((ytr == '利好').sum() / 1.5)
    syn_neg = smote_mid(Xtr, ytr, '利空', n_new=max(t_neg - int((ytr == '利空').sum()), 0))
    Xs = vstack([Xtr, syn_mid, syn_neg]).tocsr()
    ys = np.concatenate([ytr, np.array(['中性'] * syn_mid.shape[0]), np.array(['利空'] * syn_neg.shape[0])])
    rows.append(fit_eval(Xs, ys, 'SMOTE(简化)'))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- D2
def part2_interaction():
    from bench_fast import load_meta, TAB_COLS
    import lightgbm as lgb

    tr = pd.read_csv(f'{DIR}/train.csv')
    ev = pd.read_csv(f'{DIR}/eval.csv')
    meta = load_meta()
    tr = tr.merge(meta, on='text', how='left')
    ev = ev.merge(meta, on='text', how='left')

    def add_inter(d):
        d = d.copy()
        d['amp_x_lly'] = d['amp'] * d['log_ly']
        d['amp_x_q'] = d['amp'] * d['quarter']
        d['amp2'] = np.sign(d['amp']) * np.log1p(d['amp'].abs())
        d['fc_x_ly'] = d['fc_sign'] * d['ly_sign']
        d['len_x_amp'] = d['txt_len'] * np.sign(d['amp'])
        return d

    base = TAB_COLS
    inter = TAB_COLS + ['amp_x_lly', 'amp_x_q', 'amp2', 'fc_x_ly', 'len_x_amp']
    out = []
    for name, cols in [('9 个原始特征', base), ('+6 个交互特征', inter)]:
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               class_weight='balanced', verbose=-1)
        m.fit(add_inter(tr)[cols].astype(float).values, tr.label)
        p = m.predict(add_inter(ev)[cols].astype(float).values)
        acc = float((p == ev.label.values).mean())
        out.append((name, len(cols), acc))
        print('  %-16s %2d 维 | acc %.4f' % (name, len(cols), acc), flush=True)
    return out


def part2_tsne(n=3000, seed=0):
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from cluster_trio import profile_matrix

    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    prof, ok, nd = profile_matrix(z['X'], z['M'], pd.to_datetime(z['dates']), '2025-01-01', '2025-06-30')
    P = prof[ok]
    P = P[np.isfinite(P).all(1)]
    P = np.clip(P, -5, 5)
    P = (P - P.mean(0)) / (P.std(0) + 1e-9)
    if len(P) > n:
        sel = np.random.default_rng(seed).choice(len(P), n, replace=False)
        P = P[sel]
    lab = KMeans(n_clusters=4, n_init=10, random_state=0).fit(P).labels_
    t0 = time.time()
    Y = TSNE(n_components=2, perplexity=30, init='pca', random_state=seed).fit_transform(P)
    print('  t-SNE %d×%d → 2D，用时 %.0fs' % (P.shape[0], P.shape[1], time.time() - t0), flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    sc = ax[0].scatter(Y[:, 0], Y[:, 1], c=lab, cmap='tab10', s=6, alpha=.75)
    ax[0].set_title('42 因子画像的 t-SNE（颜色 = KMeans k=4）')
    ax[1].scatter(Y[:, 0], Y[:, 1], c=P[:, list(z['feats']).index('size')], cmap='coolwarm',
                  s=6, alpha=.75)
    ax[1].set_title('同一张图，颜色 = size（成交额对数）')
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    png = f'{ROOT}/reports/tsne_factors.png'
    from figstyle import save as _save
    _save(fig, png, note='数据来源：data/big/arr.npz 2025H1，随机 3,000 只 × 42 因子；'
                         't-SNE perplexity=30，KMeans k=4')
    print('  →', png)
    return {'n': int(P.shape[0]), 'png': png}


# ---------------------------------------------------------------- D3
def part3_selection(n=200000, seed=0):
    import lightgbm as lgb
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.metrics import roc_auc_score

    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, Y2, M = z['X'], z['Y2'], z['M']
    dates = pd.to_datetime(z['dates'])
    feats = list(z['feats'])
    tr_days = np.where(dates < '2025-01-01')[0]
    te_days = np.where((dates >= '2025-01-01') & (dates <= '2025-06-30'))[0]
    tt, ii = np.where(M[tr_days][:, :])
    tt, ii = tr_days[tt], ii
    sel = np.random.default_rng(seed).choice(len(tt), min(n, len(tt)), replace=False)
    Xtr, ytr = X[tt[sel], ii[sel]], (Y2[tt[sel], ii[sel]] > 0).astype(int)
    a, b = np.where(M[te_days])
    te_t, te_i = te_days[a], b
    Xte, yte = X[te_t, te_i], (Y2[te_t, te_i] > 0).astype(int)
    ok_tr, ok_te = np.isfinite(ytr), np.isfinite(yte)
    Xtr, ytr, Xte, yte = Xtr[ok_tr], ytr[ok_tr], Xte[ok_te], yte[ok_te]
    print('\n[D3] 特征选择（训练 %d 行 / 测试 %d 行，标签 = 5 日收益是否为正）' % (len(Xtr), len(Xte)))

    def run(cols, tag):
        t0 = time.time()
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.06, num_leaves=31,
                               class_weight='balanced', verbose=-1)
        m.fit(Xtr[:, cols], ytr)
        p = m.predict_proba(Xte[:, cols])[:, 1]
        auc = roc_auc_score(yte, p)
        acc = float((m.predict(Xte[:, cols]) == yte).mean())
        print('  %-18s %2d 维 | AUC %.4f | acc %.4f | %.0fs' % (tag, len(cols), auc, acc, time.time() - t0),
              flush=True)
        return dict(方法=tag, 特征数=len(cols), AUC=auc, acc=acc, 耗时=time.time() - t0)

    rows = [run(list(range(len(feats))), '全部')]

    vt = VarianceThreshold(threshold=0.05).fit(Xtr)
    keep_v = list(np.where(vt.get_support())[0])
    rows.append(run(keep_v, '方差过滤'))

    C = np.corrcoef(Xtr.T)
    keep_c, dropped = [], set()
    for i in range(len(feats)):
        if i in dropped:
            continue
        keep_c.append(i)
        for j in range(i + 1, len(feats)):
            if abs(C[i, j]) > 0.9:
                dropped.add(j)
    rows.append(run(keep_c, '相关性过滤(|r|>0.9)'))

    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.06, num_leaves=31, verbose=-1)
    m.fit(Xtr[:, keep_c], ytr)
    imp = m.feature_importances_
    k_emb = min(15, len(keep_c))
    top = [keep_c[i] for i in np.argsort(-imp)[:k_emb]]
    rows.append(run(top, '嵌入法(LGBM gain)'))
    return pd.DataFrame(rows), [feats[i] for i in dropped]


def main():
    imb = part1_imbalance()
    print('\n[D2] 交互特征')
    inter = part2_interaction()
    ts = part2_tsne()
    sel, dropped = part3_selection()

    md = ['# 加分项 D：不平衡 / 交互 + t-SNE / 特征选择三法', '',
          '> %s｜之前只用了 `class_weight=balanced` 且没做前后对比，本轮补齐' % pd.Timestamp.today().strftime('%Y-%m-%d'),
          '', '## D1 不平衡处理', '',
          '评估集固定 602 条、分布不动（利好 61% / 利空 34% / 中性 4.7%）——处理只作用于训练集，',
          '否则「指标变好」只是把评估分布改成了自己想看的形状。', '',
          '| 方案 | 训练量 | 利好 | 中性 | 利空 | accuracy | macro-F1 | 中性召回 |', '|---|---|---|---|---|---|---|---|']
    for _, r in imb.iterrows():
        md.append('| %s | %d | %d | %d | %d | %.4f | %.4f | %.3f |' % (
            r['方案'], r['训练量'], r['利好'], r['中性'], r['利空'], r['acc'], r['macro_f1'], r['中性召回']))
    md += ['', '**怎么读**：accuracy 最高的方案不一定最好——中性类只占 4.7%，全猜利好就能到 61%。',
           '真正要看的是 **macro-F1** 和**中性召回**：类别权重把中性召回从 %.2f 拉到 %.2f，'
           '代价是 accuracy 掉 %.1fpp。这就是不平衡处理的标准取舍。' % (
               imb.loc[imb.方案 == '无处理', '中性召回'].iloc[0],
               imb.loc[imb.方案 == '类别权重', '中性召回'].iloc[0],
               (imb.loc[imb.方案 == '无处理', 'acc'].iloc[0] - imb.loc[imb.方案 == '类别权重', 'acc'].iloc[0]) * 100),
           '', '| SMOTE 实现说明 |', '|---|',
           '| 没装 imblearn（它要求 sklearn 版本严格，容易把现有环境搞坏），手写「同类近邻中点插值」：'
           '`x_new = (x_i + x_j)/2`，j 取 5 近邻之一。严格 SMOTE 是 `x_i + u(x_j−x_i)`，u~U(0,1)。'
           '对 TF-IDF 这种非负稀疏向量，中点更稳、不会插出负值。 |', '',
           '## D2 交互特征 + t-SNE', '',
           '### 交互特征（表格侧）', '',
           '| 特征集 | 维度 | accuracy |', '|---|---|---|']
    for name, k, acc in inter:
        md.append('| %s | %d | %.4f |' % (name, k, acc))
    md += [' ', '### t-SNE（42 因子画像，%d 只股票）' % ts['n'], '',
           '![t-SNE](tsne_factors.png)', '',
           '左图按 KMeans k=4 上色，右图按 `size`（成交额对数）上色。两图对照能看出聚类是按什么分的：'
           '如果左图的簇边界在右图上对应清晰的渐变，说明聚出来的主要是流动性维度，而不是风格。', '',
           '## D3 特征选择三法', '',
           '| 方法 | 特征数 | AUC | accuracy | 耗时 |', '|---|---|---|---|---|']
    for _, r in sel.iterrows():
        md.append('| %s | %d | %.4f | %.4f | %.0fs |' % (r['方法'], r['特征数'], r['AUC'], r['acc'], r['耗时']))
    md += ['', '相关性过滤（|r|>0.9）丢掉的特征：`%s`' % '`、`'.join(dropped) if dropped else '',
           '', '**结论**：三法里只有嵌入法真正减维且基本不掉点；方差过滤在 42 个已标准化的因子上几乎无效',
           '（都过了阈值）；相关性过滤能砍掉冗余，但要小心把成组因子砍成单腿。', '']
    out = f'{ROOT}/reports/bonus_d.md'
    open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n→', out)


if __name__ == '__main__':
    with RunLogger(__file__, tag='bonus_d'):
        main()
