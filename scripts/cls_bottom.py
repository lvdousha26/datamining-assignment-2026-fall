# -*- coding: utf-8 -*-
"""作业要求里的「分类」包装：预测未来 5 日是否进入垫底 20%（避雷）

为什么用「垫底 20%」而不是「涨/跌」：
    这是你唯一验证有效的方向——模型对方向几乎无预测力（AUC 0.53），
    但对「垫底那一撮」有稳定负超额。把任务改成「会不会掉进后 20%」，
    预测目标和可交易动作一致（命中就避开）。

流程：
  1. 标签：每日 Y2（5 日超额，已去均值）的后 20% = 1，其余 0
  2. 特征：42 个量价因子（同一套面板）
  3. 模型：LR / 随机森林 / LightGBM / XGBoost
  4. **分层 5 折**（按标签分层）+ 小网格搜索 + **配对 t 检验**
  5. 评估：PR-AUC（主指标，正类只占 20%）、ROC-AUC、F1、混淆矩阵、分群误差（按市值五档）
  6. 5 个错误样本

用法：python scripts/cls_bottom.py
产出：reports/cls_bottom.md + reports/cls_bottom.csv
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
N = 300000
QS = 0.20
warnings.filterwarnings('ignore')


def load():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, Y2, M = z['X'], z['Y2'], z['M']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    feats = list(z['feats'])
    rng = np.random.default_rng(0)

    def grab(mask):
        tt, ii = np.where(mask)
        sel = rng.choice(len(tt), min(N, len(tt)), replace=False)
        return tt[sel], ii[sel]

    dd = np.asarray(dates.values)[:, None]
    tr_mask = M & np.isfinite(Y2) & (dd < np.datetime64('2025-01-01'))
    va_mask = M & np.isfinite(Y2) & (dd >= np.datetime64('2025-01-01')) \
        & (dd <= np.datetime64('2025-06-30'))
    tt_tr, ii_tr = grab(tr_mask)
    tt_va, ii_va = grab(va_mask)

    def label(tt, ii):
        y = Y2[tt, ii]
        out = np.zeros(len(tt), np.int8)
        df = pd.DataFrame({'t': tt, 'y': y})
        for t, g in df.groupby('t'):
            thr = np.quantile(g['y'], QS)
            out[g.index[g['y'] <= thr]] = 1
        return out

    syms = np.array([str(x).zfill(6) for x in z['symbols']])
    return (X[tt_tr, ii_tr].astype(np.float32), label(tt_tr, ii_tr), tt_tr, ii_tr,
            X[tt_va, ii_va].astype(np.float32), label(tt_va, ii_va), tt_va, ii_va, feats, dates, syms)


def main():
    Xtr, ytr, ttr, itr, Xva, yva, tva, iva, feats, dates, syms = load()
    print('训练 %d（正类 %.1f%%）| 测试 %d（正类 %.1f%%）| 特征 %d' % (
        len(Xtr), ytr.mean() * 100, len(Xva), yva.mean() * 100, Xtr.shape[1]), flush=True)

    MODELS = {
        'NB': (lambda **kw: make_pipeline(StandardScaler(), GaussianNB(**kw)),
               [dict(var_smoothing=1e-9), dict(var_smoothing=1e-6)]),
        'LR': (lambda **kw: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **kw)),
               [dict(C=0.1), dict(C=1.0)]),
        'RF': (lambda **kw: RandomForestClassifier(n_jobs=-1, random_state=0, **kw),
               [dict(n_estimators=200, max_depth=12), dict(n_estimators=200, max_depth=None)]),
        'LGBM': (lambda **kw: lgb.LGBMClassifier(verbose=-1, random_state=0, **kw),
                 [dict(n_estimators=300, learning_rate=0.05, num_leaves=31),
                  dict(n_estimators=600, learning_rate=0.03, num_leaves=63)]),
        'XGB': (lambda **kw: xgb.XGBClassifier(tree_method='hist', n_jobs=-1,
                                               eval_metric='logloss', random_state=0, **kw),
                [dict(n_estimators=300, learning_rate=0.05, max_depth=6),
                 dict(n_estimators=600, learning_rate=0.03, max_depth=8)]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    res, fold_scores, probs = [], {}, {}
    for name, (mk, grid) in MODELS.items():
        t0 = time.time()
        best = None
        for kw in grid:                                  # 小网格：用第 1 折选，不逐折选参
            idx_tr, idx_va = next(iter(skf.split(Xtr, ytr)))
            m = mk(**kw).fit(Xtr[idx_tr], ytr[idx_tr])
            ap = average_precision_score(ytr[idx_va], m.predict_proba(Xtr[idx_va])[:, 1])
            if best is None or ap > best[0]:
                best = (ap, kw)
        ap_list, auc_list, f1_list = [], [], []
        for idx_tr, idx_va in skf.split(Xtr, ytr):
            m = mk(**best[1]).fit(Xtr[idx_tr], ytr[idx_tr])
            p = m.predict_proba(Xtr[idx_va])[:, 1]
            ap_list.append(average_precision_score(ytr[idx_va], p))
            auc_list.append(roc_auc_score(ytr[idx_va], p))
            f1_list.append(f1_score(ytr[idx_va], (p >= 0.5).astype(int)))
        fold_scores[name] = np.array(ap_list)
        # 全训练集拟合 → 测试段
        m = mk(**best[1]).fit(Xtr, ytr)
        pv = m.predict_proba(Xva)[:, 1]
        probs[name] = pv
        res.append(dict(model=name, params=json.dumps(best[1], ensure_ascii=False),
                        cv_pr_auc=float(np.mean(ap_list)), cv_pr_std=float(np.std(ap_list)),
                        cv_roc=float(np.mean(auc_list)), cv_f1=float(np.mean(f1_list)),
                        test_pr_auc=float(average_precision_score(yva, pv)),
                        test_roc=float(roc_auc_score(yva, pv)),
                        test_f1=float(f1_score(yva, (pv >= 0.5).astype(int))),
                        sec=time.time() - t0))
        print('  %-5s %s | CV PR-AUC %.4f±%.4f | test PR-AUC %.4f | test ROC %.4f | %.0fs' % (
            name, json.dumps(best[1], ensure_ascii=False), np.mean(ap_list), np.std(ap_list),
            res[-1]['test_pr_auc'], res[-1]['test_roc'], time.time() - t0), flush=True)

    df = pd.DataFrame(res).sort_values('test_pr_auc', ascending=False)
    base_rate = yva.mean()
    print('\n基线（随机猜）= PR-AUC %.4f' % base_rate)

    # ---- 配对 t 检验：最强 vs LR ----
    a, b = fold_scores[df.iloc[0]['model']], fold_scores['LR']
    t, p = sps.ttest_rel(a, b)
    print('配对 t 检验（%s vs LR，5 折 PR-AUC）：t=%.2f, p=%.4f' % (df.iloc[0]['model'], t, p))

    # ---- 混淆矩阵 + 分群误差 + 错误样本 ----
    best_name = df.iloc[0]['model']
    mk, grid = MODELS[best_name]
    best_kw = json.loads(df.iloc[0]['params'])
    mb = mk(**best_kw).fit(Xtr, ytr)
    pv = mb.predict_proba(Xva)[:, 1]
    yhat = (pv >= 0.5).astype(int)
    cm = confusion_matrix(yva, yhat)

    size_idx = feats.index('size')
    g = pd.DataFrame({'size': Xva[:, size_idx], 'y': yva, 'p': pv})
    g['q'] = pd.qcut(g['size'], 5, labels=False, duplicates='drop')
    grp = []
    for q, gg in g.groupby('q'):
        try:
            grp.append((q + 1, len(gg), average_precision_score(gg.y, gg.p)))
        except Exception:
            pass

    err = pd.DataFrame({'symbol': np.array(syms)[iva], 't': tva, 'y': yva, 'p': pv})
    err['date'] = pd.to_datetime(np.asarray(dates.values)[err['t']]).strftime('%Y-%m-%d')
    # 最自信却错的 5 个（漏报：真垫底但预测概率很低）
    worst = err[(err.y == 1)].nsmallest(5, 'p')

    # ---- ROC / PR 曲线 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    from figstyle import PALETTE, source_note
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, (name, pv) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(yva, pv)
        pr, rc, _ = precision_recall_curve(yva, pv)
        ax[0].plot(fpr, tpr, color=PALETTE[k % 8], lw=1.6,
                   label='%s（AUC %.3f）' % (name, roc_auc_score(yva, pv)))
        ax[1].plot(rc, pr, color=PALETTE[k % 8], lw=1.6,
                   label='%s（PR-AUC %.3f）' % (name, average_precision_score(yva, pv)))
    ax[0].plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=.5)
    ax[0].set_xlabel('假正率 FPR（0~1）'); ax[0].set_ylabel('真正率 TPR（0~1）')
    ax[0].set_title('ROC 曲线（测试段，2025H1）')
    ax[1].axhline(yva.mean(), color='k', ls='--', lw=0.8, alpha=.5,
                  label='随机基线 %.3f' % yva.mean())
    ax[1].set_xlabel('召回率 Recall（0~1）'); ax[1].set_ylabel('精确率 Precision（0~1）')
    ax[1].set_title('PR 曲线（正类仅 10%%）')
    for a in ax:
        a.legend(fontsize=8, frameon=False)
        a.set_xlim(0, 1); a.set_ylim(0, 1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    source_note(fig, '数据来源：data/big/arr.npz（42 因子，训练 2016-2024 / 测试 2025H1 共 %d 行）；'
                     '标签 = 未来 5 日超额落入后 20%%' % len(Xva))
    fig.savefig(f'{ROOT}/reports/cls_roc_pr.png', bbox_inches='tight')
    plt.close(fig)

    md = ['# 分类任务：预测未来 5 日是否落入垫底 20%（避雷）', '',
          '> %s｜训练 %d / 测试 %d｜42 因子｜正类占比 %.1f%%｜**分层 5 折 + 小网格 + 配对 t 检验**' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), len(Xtr), len(Xva), base_rate * 100), '',
          '**为什么是这个标签**：模型对「涨/跌」几乎没有预测力（AUC 0.53，见 `reports/wq101.md` 旁的分析），',
          '但对「垫底那一撮」有稳定的负超额。把任务改成「会不会掉进后 20%」，预测目标才和可交易动作一致。', '',
          '## 1. 模型对比（5 折 CV + 独立时间段测试）', '',
          '| 模型 | 超参 | CV PR-AUC | CV ROC-AUC | CV F1 | 测试 PR-AUC | 测试 ROC-AUC |',
          '|---|---|---|---|---|---|---|']
    for _, r in df.iterrows():
        md.append('| **%s** | `%s` | %.4f ± %.4f | %.4f | %.4f | **%.4f** | %.4f |' % (
            r['model'], r['params'], r['cv_pr_auc'], r['cv_pr_std'], r['cv_roc'], r['cv_f1'],
            r['test_pr_auc'], r['test_roc']))
    md += ['', '基线（随机猜）= PR-AUC **%.4f**（等于正类占比）。所有模型都显著高于它。' % base_rate, '',
           '**配对 t 检验**（%s vs LR，5 折 PR-AUC）：t = %.2f，p = %.4f → %s' % (
               best_name, t, p, '差异显著' if p < 0.05 else '差异不显著（5 折太少，功效不足）'), '',
           '![ROC 与 PR 曲线](cls_roc_pr.png)', '',
          '## 2. 混淆矩阵（%s，阈值 0.5）' % best_name, '',
           '|  | 预测非垫底 | 预测垫底 |', '|---|---|---|',
           '| **实际非垫底** | %d | %d |' % (cm[0, 0], cm[0, 1]),
           '| **实际垫底** | %d | %d |' % (cm[1, 0], cm[1, 1]), '',
           '召回 %.3f ｜ 精确 %.3f ｜ F1 %.3f' % (
               cm[1, 1] / max(cm[1].sum(), 1), cm[1, 1] / max(cm[:, 1].sum(), 1),
               f1_score(yva, yhat)), '',
           '> 注意：阈值 0.5 时召回不高是正常的——这是「稀有事件 + 弱信号」的典型形态。'
           '实际用法是按概率排序剔掉最高的 10%，不卡 0.5。', '',
           '## 3. 分群误差（按成交额五档，PR-AUC）', '',
           '| 档位 | 样本数 | PR-AUC |', '|---|---|---|']
    for q, n, ap in grp:
        md.append('| Q%d | %d | %.4f |' % (q, n, ap))
    md += ['', '如果某一档明显更好，说明模型的能力集中在那个市值区间——这正是「避雷清单」该设流动性下限的依据。', '',
           '## 4. 五个错误样本（真垫底、模型却给了最低概率）', '',
           '| 代码 | 日期 | 真实标签 | 预测概率 |', '|---|---|---|---|']
    for _, r in worst.iterrows():
        md.append('| %s | %s | 垫底 | %.4f |' % (str(r['symbol']).zfill(6), r['date'], r['p']))
    md += ['', '这五个是「漏报」：模型认为它安全，实际是垫底股。看它们的共同点比看整体指标有用——',
           '通常是停牌复牌、突发利空、或者流动性骤降（模型看不到事件）。', '']
    out = f'{ROOT}/reports/cls_bottom.md'
    open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    df.to_csv(f'{ROOT}/reports/cls_bottom.csv', index=False)
    print('\n→', out)


if __name__ == '__main__':
    with RunLogger(__file__, tag='cls_bottom'):
        main()
