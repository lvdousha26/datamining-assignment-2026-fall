# -*- coding: utf-8 -*-
"""「即将被调出两融」事前预警因子

思路
----
调出两融当天再动作已经晚了（事件前 20 日已经跌 10.6%，见 `reports/ev_margin.md`）。
所以把问题前移：**在 T 日，用 T 日及之前的量价 + 两融余额特征，预测未来 5 个交易日内会不会被调出。**

特征（都是「出事」的直接表征）
  exc20       过去 20 日相对全市场等权的超额（跌得凶）
  ret5        过去 5 日收益
  turn5       5 日换手
  turn_ratio  5 日/60 日换手比（换手突变）
  vol_amp     std20/std60（波动放大）
  amp20       20 日振幅
  size        成交额对数
  illiq20     非流动性
  bal_ratio   融资余额 / 20 日日均成交额（杠杆依赖度）

标签：T 日在两融名单里，且 5 个交易日内被调出 → 1；否则 0。

切分：2019-2023 训练 / 2024-2026 测试（严格按时间，不随机）。

用法：python scripts/margin_alert.py
产出：reports/margin_alert.md + csv
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
STEP = 5
HORIZON = 5


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, M = z['X'], z['M']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    feats = list(z['feats'])
    syms = [str(s).zfill(6) for s in z['symbols']]
    col = {s: j for j, s in enumerate(syms)}
    Fi = {f: i for i, f in enumerate(feats)}

    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    R = d['R']
    T = len(dates)
    # 过去 20 日超额（不含当日之后）
    Rf = np.nan_to_num(R, nan=0.0)
    cs = np.cumsum(np.log1p(Rf), 0)
    pad = np.zeros((20, R.shape[1]), np.float32)
    past20 = np.expm1(cs - np.vstack([pad, cs[:-20]]))
    ew20 = np.expm1(np.cumsum(np.log1p(np.nan_to_num(np.nanmean(R, 1), nan=0.0))) -
                    np.concatenate([np.zeros(20), np.cumsum(np.log1p(np.nan_to_num(np.nanmean(R, 1), nan=0.0)))[:-20]]))
    ex20 = (1 + past20) / (1 + ew20[:, None]) - 1

    # 成员名单（按抽样日）
    mem = {}
    for f in glob.glob(f'{ROOT}/data/big/margin/*.parquet'):
        dd = os.path.basename(f)[:10]
        mem[dd] = set(pd.read_parquet(f)['symbol'].astype(str).str.zfill(6))
    # 融资余额
    bal = {}
    for f in glob.glob(f'{ROOT}/data/big/margin_bal/*.parquet'):
        bal[os.path.basename(f)[:10]] = pd.read_parquet(f).set_index('symbol')['bal'].to_dict()

    ev = pd.read_csv(f'{ROOT}/events/margin_del.csv', dtype={'symbol': str})
    ev['date'] = pd.to_datetime(ev['date'])
    di = {pd.Timestamp(x).date(): i for i, x in enumerate(dates)}
    ev_idx = {}
    for _, r in ev.iterrows():
        j = di.get(r['date'].date())
        if j is not None:
            ev_idx.setdefault(j, set()).add(r['symbol'])

    rows = []
    used_days = [x for x in sorted(mem) if pd.Timestamp(x) >= pd.Timestamp('2019-01-01')][::STEP]
    for ds in used_days:
        j = di.get(pd.Timestamp(ds).date())
        if j is None or j < 60 or j + HORIZON >= T:
            continue
        ms = mem[ds]
        idx = np.array([col[s] for s in ms if s in col])
        if len(idx) < 300:
            continue
        near = set()
        for k in range(1, HORIZON + 1):
            near |= ev_idx.get(j + k, set())
        amt20 = np.expm1(X[j, idx, Fi['size']]) + 1
        bal_map = bal.get(ds, {})
        b = np.array([bal_map.get(syms[i], np.nan) for i in idx], dtype=np.float32)
        F = np.column_stack([
            ex20[j, idx], X[j, idx, Fi['ret5']], X[j, idx, Fi['turn5']],
            X[j, idx, Fi['turn_ratio']], X[j, idx, Fi['std20']] / (X[j, idx, Fi['std60']] + 1e-6),
            X[j, idx, Fi['amp20']], X[j, idx, Fi['size']], X[j, idx, Fi['illiq20']],
            b / amt20,
        ])
        y = np.array([1 if syms[i] in near else 0 for i in idx], dtype=np.int8)
        rows.append((ds, j, idx, F, y))

    print('可用期数 %d' % len(rows), flush=True)
    Atr = [r for r in rows if pd.Timestamp(r[0]) < pd.Timestamp('2024-01-01')]
    Ate = [r for r in rows if pd.Timestamp(r[0]) >= pd.Timestamp('2024-01-01')]
    Xtr = np.vstack([r[3] for r in Atr]); ytr = np.concatenate([r[4] for r in Atr])
    Xte = np.vstack([r[3] for r in Ate]); yte = np.concatenate([r[4] for r in Ate])
    print('训练 %d 期 / %d 样本（正类 %d, %.3f%%）| 测试 %d 期 / %d 样本（正类 %d, %.3f%%）' % (
        len(Atr), len(Xtr), ytr.sum(), ytr.mean() * 100, len(Ate), len(Xte), yte.sum(), yte.mean() * 100),
        flush=True)

    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                           min_data_in_leaf=100, subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=10,
                           verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    ap = average_precision_score(yte, p)
    auc = roc_auc_score(yte, p)
    base = yte.mean()
    print('\n测试：PR-AUC %.4f（基线 %.4f，%.0f×）| ROC-AUC %.4f' % (
        ap, base, ap / max(base, 1e-9), auc), flush=True)

    # 逐期 top-k 精度（K 取当期标的数的 1%，避免 20 只在 3300 只里太极端）
    K_PCT = 0.01
    hits, rand = [], []
    off = 0
    for ds, j, idx, F, y in Ate:
        n = len(idx)
        pp = p[off:off + n]; off += n
        K = max(int(n * K_PCT), 10)
        top = np.argsort(-pp)[:K]
        hits.append(y[top].mean())
        rand.append(y.mean())       # 随机抽 K 个的期望命中率 = 当期正类占比
    hits = np.array(hits); rand = np.array(rand)
    pd.DataFrame({'date': [r[0] for r in Ate], 'n': [len(r[2]) for r in Ate],
                  'pos': [int(r[4].sum()) for r in Ate], 'top_hit': hits}).to_csv(
        f'{ROOT}/reports/margin_alert_periods.csv', index=False)
    np.save(f'{ROOT}/runs/laya_yjyg/margin_pred.npy', p)
    np.save(f'{ROOT}/runs/laya_yjyg/margin_y.npy', yte)
    print('每期 top-1%% 命中率 %.2f%%（基线 %.2f%%，%.1f×）' % (
        hits.mean() * 100, rand.mean() * 100, hits.mean() / max(rand.mean(), 1e-9)), flush=True)

    imp = pd.Series(m.feature_importances_, index=['exc20', 'ret5', 'turn5', 'turn_ratio',
                                                   'vol_amp', 'amp20', 'size', 'illiq20', 'bal_ratio'])
    imp = imp.sort_values(ascending=False)

    md = ['# 事前预警：预测「即将被调出两融」', '',
          '> %s｜训练 2019-2023（%d 样本）/ 测试 2024-2026（%d 样本）｜正类占比 %.3f%%' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), len(Xtr), len(Xte), base * 100), '',
          '**为什么前移**：调出当天再动手已经晚了——事件前 20 日平均已跌 10.6%（`reports/ev_margin.md`）。',
          '这里把任务改成「未来 5 个交易日会不会被调出」。', '',
          '## 结果', '',
          '| 指标 | 值 | 基线 | 倍数 |', '|---|---|---|---|',
          '| PR-AUC | **%.4f** | %.5f | **%.0f×** |' % (ap, base, ap / max(base, 1e-9)),
          '| ROC-AUC | %.4f | 0.5 | — |' % auc,
          '| 每期 top-1%% 命中率 | **%.2f%%** | %.2f%% | %.1f× |' % (
              hits.mean() * 100, rand.mean() * 100, hits.mean() / max(rand.mean(), 1e-9)),
          '', '## 特征重要性', '', '| 特征 | 重要性 |', '|---|---|']
    for k, v in imp.items():
        md.append('| %s | %d |' % (k, v))
    md += ['', '## 怎么用', '',
           '1. **不是用来做空**（A 股无券），而是持仓风险预警：分数进前 20 的持仓，5 日内被调出两融的概率',
           '   是随机抽 20 只的 %.0f 倍，值得提前减仓；' % (hits.mean() / max(rand.mean(), 1e-9)),
           '2. **和禁买清单互补**：D1 抓量价异常，这个抓「杠杆资金要撤」，机制不同；',
           '3. 上线前必须在 `purged_cv.py` 协议下测组合层面年化——命中率不等于钱（刚被这条教训过一次）。', '',
           '## 局限', '',
           '- 正类只有 %d 个（测试段），PR-AUC 的置信区间很宽，±30%% 不奇怪；' % int(yte.sum()),
           '- 只按时间切了一次，没做多折；',
           '- `bal_ratio` 依赖融资余额，只在抽样日抓（每 5 日），日频部署要补全；',
           '- 特征全是量价 + 余额，没有事件信息（ST、违规公告），而调出常常由这些触发。', '']
    open(f'{ROOT}/reports/margin_alert.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    json.dump({'pr_auc': float(ap), 'roc_auc': float(auc), 'base': float(base),
               'top20_hit': float(hits.mean()), 'top20_rand': float(rand.mean()),
               'n_train': int(len(Xtr)), 'n_test': int(len(Xte)),
               'imp': {k: int(v) for k, v in imp.items()}},
              open(f'{ROOT}/reports/margin_alert.json', 'w'), ensure_ascii=False, indent=1)
    print('\n特征重要性：\n' + imp.to_string())
    print('→', f'{ROOT}/reports/margin_alert.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='margin_alert'):
        main()
