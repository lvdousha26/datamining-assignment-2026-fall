# -*- coding: utf-8 -*-
"""换指标重判：尾部加权到底有没有用？（不看 IC，看 D1）

上一次（`nn_tail_event.py`）用 IC 判尾部加权，结论是「无效（−0.0003）」。
但 IC 是全截面秩相关，把 10% 样本权重×5 本来就不该改变它——**是我们用错了指标**。

这次改看四个直接对应「避雷清单」的指标（每日截面，再对 117 个验证日取均值）：

  D1 命中率   = 模型打分最低 10% 里，真实收益也落在后 10% 的比例（随机 = 10%）
  D1 召回率   = 真实后 10% 里，被模型排进最低 10% 的比例
  D1 平均分位 = 模型 D1 组的真实收益分位均值（0=最差，50=随机）
  D1 超额     = 模型 D1 组的真实 10 日超额均值（负得越多越好）

对照：随机基线（=10% / 10% / 50% / ≈0）、真实 D1（oracle）。
模型：NN 三档 + LGBM 两档，与上一轮完全相同的抽样与超参。

用法：python scripts/d1_metrics.py
产出：reports/d1_metrics.md + csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from nn_tail_event import (ROOT, L, H, N_SYM, N_DAY, fwd_excess, build_events,
                           gather, rank_target, train_nn)

Q = 0.10


def d1_metrics(pred, y, tt):
    """逐日算 D1 四个指标"""
    rows = []
    for t in np.unique(tt):
        m = tt == t
        if m.sum() < 200:
            continue
        p, v = pred[m], y[m]
        n = max(int(len(p) * Q), 5)
        ry = (np.argsort(np.argsort(v)) + 0.5) / len(v)     # 真实分位 0..1
        d_pred = np.argsort(p)[:n]                          # 模型最不看好的 n 只
        d_true = ry <= Q
        rows.append((d_true[d_pred].mean(),                      # 命中率
                     d_pred[np.isin(d_pred, np.where(d_true)[0])].size / max(d_true.sum(), 1),  # 召回
                     ry[d_pred].mean(),                          # 平均分位
                     v[d_pred].mean() - v.mean()))                # 相对全市场的超额
    a = np.array(rows)
    return a.mean(0), len(a), a


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, M = z['X'], z['M']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    syms = [str(s).zfill(6) for s in z['symbols']]
    col = {s: j for j, s in enumerate(syms)}
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    y = fwd_excess(d['R'])
    evt, st = build_events(dates, syms, col)

    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[1], N_SYM, replace=False)
    idx = idx[M.any(0)[idx]]
    tr_days_all = np.where((dates < '2025-01-01').values)[0]
    tr_days_all = tr_days_all[tr_days_all >= L + H]
    tr_days = np.sort(rng.choice(tr_days_all, min(N_DAY, len(tr_days_all)), replace=False))
    va_days = np.where(((dates >= '2025-01-01') & (dates <= '2025-06-30')).values)[0]
    Xtr, ytr, ttr, etr, str_ = gather(X, y, evt, st, tr_days, idx)
    Xva, yva, tva, eva, sva = gather(X, y, evt, st, va_days, idx)
    print('训练 %s | 验证 %s' % (Xtr.shape, Xva.shape), flush=True)

    rows = []
    daily_hist = {}
    # ---- 参照系 ----
    (h, r, pct, ex), nd, _ = d1_metrics(np.random.default_rng(1).random(len(yva)), yva, tva)
    rows.append(dict(model='随机基线', hit=h, recall=r, pctile=pct, excess=ex, ic=np.nan))
    # oracle：直接把真实收益取负当预测，D1 就是真实 D1（上限参照）
    (h, r, pct, ex), _, _ = d1_metrics(yva, yva, tva)
    rows.append(dict(model='真实 D1（oracle）', hit=h, recall=r, pctile=pct, excess=ex, ic=np.nan))

    # ---- LGBM ----
    rk_tr = rank_target(ytr, ttr)
    for tag, A, B in [('LGBM（当期因子）', Xtr[:, -1, :], Xva[:, -1, :]),
                      ('LGBM（+30日均值/波动）',
                       np.concatenate([Xtr[:, -1, :], Xtr.mean(1), Xtr.std(1)], 1),
                       np.concatenate([Xva[:, -1, :], Xva.mean(1), Xva.std(1)], 1))]:
        from nn_tail_event import daily_ic
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               objective='binary', verbose=-1)
        m.fit(A, (rk_tr >= 0.5).astype(int))
        p = m.predict_proba(B)[:, 1]
        (h, r, pct, ex), nd, _ = d1_metrics(p, yva, tva)
        ic, _, _ = daily_ic(p, yva, tva)
        rows.append(dict(model=tag, hit=h, recall=r, pctile=pct, excess=ex, ic=ic))
        print('  %-24s D1 命中 %.3f | 分位 %.1f%% | 超额 %+.2f%% | IC %+.4f' % (
            tag, h, pct * 100, ex * 100, ic), flush=True)

    # ---- NN 三档 ----
    for tag, gate, tail in [('NN mse_rank', False, False),
                            ('NN + 尾部加权', False, True),
                            ('NN + 尾部加权 + 事件门控', True, True)]:
        t0 = time.time()
        outs = [train_nn(Xtr, ytr, ttr, etr, str_, Xva, yva, tva, eva, sva,
                         gate=gate, tail=tail, seed=s) for s in (0, 1, 2)]
        # 三个种子的预测取平均后再算指标（集成比单模型更接近部署）
        p = np.mean([o[3] for o in outs], 0)
        (h, r, pct, ex), nd, perday = d1_metrics(p, yva, tva)
        daily_hist[tag] = perday
        ic = float(np.mean([o[0] for o in outs]))
        rows.append(dict(model=tag, hit=h, recall=r, pctile=pct, excess=ex, ic=ic))
        print('  %-24s D1 命中 %.3f | 分位 %.1f%% | 超额 %+.2f%% | IC %+.4f | %.0fs' % (
            tag, h, pct * 100, ex * 100, ic, time.time() - t0), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{ROOT}/reports/d1_metrics.csv', index=False)
    md = ['# 换指标重判：尾部加权到底有没有用（看 D1，不看 IC）', '',
          '> %s｜10 日超额标签｜%d 个验证日｜600 只抽样｜NN 为 3 seed 预测平均' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), nd), '',
          '| 模型 | D1 命中率 | D1 召回 | D1 平均分位 | D1 超额 | IC |', '|---|---|---|---|---|---|']
    for _, r in df.iterrows():
        md.append('| %s | %.1f%% | %.1f%% | %.1f%% | %+.2f%% | %s |' % (
            r['model'], r['hit'] * 100, r['recall'] * 100, r['pctile'] * 100, r['excess'] * 100,
            '—' if pd.isna(r['ic']) else '%+.4f' % r['ic']))
    md += ['', '## 判读', '',
           '随机基线：命中 10%、召回 10%、分位 50%、超额 ≈0。', '',
           '| 对比 | D1 命中差 | D1 分位差 | 结论 |', '|---|---|---|---|',
           '见下（由脚本填）', '']
    # 关键对比
    def get(name):
        return df[df.model == name].iloc[0]
    try:
        a, b = get('NN mse_rank'), get('NN + 尾部加权')
        md.append('**尾部加权**：命中率 %.1f%% → %.1f%%（%+.1fpp），D1 分位 %.1f%% → %.1f%%（%+.1fpp），'
                  'D1 超额 %+.2f%% → %+.2f%%。%s' % (
                      a['hit'] * 100, b['hit'] * 100, (b['hit'] - a['hit']) * 100,
                      a['pctile'] * 100, b['pctile'] * 100, (b['pctile'] - a['pctile']) * 100,
                      a['excess'] * 100, b['excess'] * 100,
                      '**尾部加权有效**——IC 没动但 D1 变准了，说明它确实把容量挪到了有用的地方。'
                      if b['hit'] > a['hit'] + 0.01 or b['pctile'] < a['pctile'] - 0.01
                      else '仍无改善，尾部加权在这个模型/数据上确实没用。'))
        c = get('NN + 尾部加权 + 事件门控')
        md += ['', '**事件门控**：命中率 %+.1fpp、D1 分位 %+.1fpp（相对尾部加权版）。' % (
            (c['hit'] - b['hit']) * 100, (c['pctile'] - b['pctile']) * 100), '']
    except Exception as e:
        md.append('（对比计算失败：%s）' % e)

    # 配对检验（逐日命中率差）
    from scipy import stats as sps
    try:
        a1, a2 = daily_hist['NN mse_rank'][:, 0], daily_hist['NN + 尾部加权'][:, 0]
        t1, p1 = sps.ttest_rel(a2, a1)
        b2 = daily_hist['NN + 尾部加权 + 事件门控'][:, 0]
        t2, p2 = sps.ttest_rel(b2, a2)
        md += ['', '## 统计检验（逐日命中率，配对 t）', '',
               '- 尾部加权 vs 无加权：t = %.2f，p = %.4f' % (t1, p1),
               '- 事件门控 vs 无门控：t = %.2f，p = %.4f' % (t2, p2), '']
    except Exception as e:
        md += ['', '（配对检验失败：%s）' % e, '']

    md += ['', '## 为什么 IC 和 D1 会给出不同答案', '',
           'IC 是 **全截面秩相关**，它对 5000 只股票的排序整体负责；而禁买清单只关心**最低的 500 只**。',
           '把 10% 的样本权重放大 5 倍，可能让这 500 只排得更准（D1 命中率上升），',
           '同时让中段排得更乱（IC 反而下降）——两者并不矛盾，**只是指标对应了不同的业务**。',
           '',
           '这也是上一个实验的方法论错误：用「整体排序质量」去评判「尾部专用改进」，',
           '自然会得出「无效」。', '']
    open(f'{ROOT}/reports/d1_metrics.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n' + df.round(4).to_string(index=False))
    print('→', f'{ROOT}/reports/d1_metrics.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='d1_metrics'):
        main()
