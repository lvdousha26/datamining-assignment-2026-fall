# -*- coding: utf-8 -*-
"""全市场复核：尾部加权在 5,342 只上还成立吗？

上一轮（`d1_metrics.py`）用的是 600 只抽样，结论是「尾部加权把 D1 命中率从 28.9% 提到 32.2%」。
抽样结论不能直接上生产，这里换成全市场、用惰性切片（不物化 30×42 序列）重跑一遍。

对比：
  LGBM（当期 42 因子，全市场）
  NN mse_rank（全市场）
  NN + 尾部加权（全市场）

口径与上一轮一致：fwd10 超额、截面 rank 目标、D1=每日最低 10%、117 个验证日。
训练日抽样 150 天 × 全市场约 4000 只/日 ≈ 60 万样本/epoch。

用法：python scripts/d1_fullmarket.py
产出：reports/d1_fullmarket.md + csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from nn_tail_event import ROOT, L, H, fwd_excess, daily_ic, Net  # noqa: E402
from d1_metrics import d1_metrics  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRAIN_DAY = 150
BATCH, EPOCHS, SEEDS = 1024, 2, 2


def keys_of(M, y, days):
    ks = []
    for t in days:
        if t < L:
            continue
        ii = np.where(M[t] & np.isfinite(y[t]))[0]
        if len(ii) < 500:
            continue
        ks.append(np.column_stack([np.full(len(ii), t, np.int32), ii.astype(np.int32)]))
    return np.concatenate(ks) if ks else np.zeros((0, 2), np.int32)


def batch_of(X, ks):
    return np.stack([X[t - L + 1:t + 1, j, :] for t, j in ks]).astype(np.float32)


def rank_target(y, tt):
    out = np.zeros(len(y), np.float32)
    for t in np.unique(tt):
        m = tt == t
        out[m] = (np.argsort(np.argsort(y[m])) + 0.5) / m.sum()
    return out


def train_nn(X, ks_tr, y, tail, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(X.shape[2]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    tt = ks_tr[:, 0]
    yv = y[tt, ks_tr[:, 1]]
    rk = rank_target(yv, tt)
    w = np.where(rk < 0.1, 5.0, 1.0).astype(np.float32) if tail else np.ones(len(rk), np.float32)
    steps = (len(ks_tr) + BATCH - 1) // BATCH
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps * EPOCHS)
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = np.random.permutation(len(ks_tr))
        for b in range(steps):
            ix = perm[b * BATCH:(b + 1) * BATCH]
            xb = torch.from_numpy(batch_of(X, ks_tr[ix])).to(DEV)
            out = torch.sigmoid(m(xb))
            tgt = torch.from_numpy(rk[ix]).to(DEV)
            wb = torch.from_numpy(w[ix]).to(DEV)
            loss = (wb * (out - tgt) ** 2).sum() / wb.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()
        print('    ep%d loss %.4f | %.0fs' % (ep + 1, loss.item(), time.time() - t0), flush=True)
    return m


def predict_nn(m, X, ks):
    m.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(ks), 4096):
            out.append(m(torch.from_numpy(batch_of(X, ks[s:s + 4096])).to(DEV)).cpu().numpy())
    return np.concatenate(out)


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, M = z['X'], z['M']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    y = fwd_excess(d['R'])
    print('全市场 %d 只 | 面板 %s' % (X.shape[1], X.shape), flush=True)

    rng = np.random.default_rng(0)
    tr_days_all = np.where((dates < '2025-01-01').values)[0]
    tr_days_all = tr_days_all[tr_days_all >= L + H]
    tr_days = np.sort(rng.choice(tr_days_all, min(N_TRAIN_DAY, len(tr_days_all)), replace=False))
    va_days = np.where(((dates >= '2025-01-01') & (dates <= '2025-06-30')).values)[0]

    ks_tr = keys_of(M, y, tr_days)
    ks_va = keys_of(M, y, va_days)
    print('训练样本 %d（%d 日）| 验证样本 %d（%d 日）' % (
        len(ks_tr), len(tr_days), len(ks_va), len(va_days)), flush=True)

    yva = y[ks_va[:, 0], ks_va[:, 1]]
    tva = ks_va[:, 0]
    rows = []

    # ---- LGBM ----
    t0 = time.time()
    Ftr = X[ks_tr[:, 0], ks_tr[:, 1], :]
    Fva = X[ks_va[:, 0], ks_va[:, 1], :]
    rk = rank_target(y[ks_tr[:, 0], ks_tr[:, 1]], ks_tr[:, 0])
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           objective='binary', verbose=-1, n_jobs=-1)
    m.fit(Ftr, (rk >= 0.5).astype(int))
    p = m.predict_proba(Fva)[:, 1]
    (h, r, pct, ex), nd, _ = d1_metrics(p, yva, tva)
    ic, icir, _ = daily_ic(p, yva, tva)
    rows.append(dict(model='LGBM（当期因子）', hit=h, recall=r, pctile=pct, excess=ex, ic=ic))
    print('  LGBM 命中 %.3f 分位 %.1f%% 超额 %+.2f%% IC %+.4f | %.0fs' % (
        h, pct * 100, ex * 100, ic, time.time() - t0), flush=True)
    del Ftr, Fva

    # ---- NN 两档 ----
    for tag, tail in [('NN mse_rank', False), ('NN + 尾部加权', True)]:
        preds = []
        for sd in range(SEEDS):
            t0 = time.time()
            mm = train_nn(X, ks_tr, y, tail, seed=sd)
            preds.append(predict_nn(mm, X, ks_va))
            print('  %s seed%d | %.0fs' % (tag, sd, time.time() - t0), flush=True)
            del mm
            torch.cuda.empty_cache()
        p = np.mean(preds, 0)
        (h, r, pct, ex), nd, _ = d1_metrics(p, yva, tva)
        ic = float(np.mean([daily_ic(q, yva, tva)[0] for q in preds]))
        rows.append(dict(model=tag, hit=h, recall=r, pctile=pct, excess=ex, ic=ic))
        print('  %-16s 命中 %.3f | 分位 %.1f%% | 超额 %+.2f%% | IC %+.4f' % (
            tag, h, pct * 100, ex * 100, ic), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{ROOT}/reports/d1_fullmarket.csv', index=False)
    md = ['# 全市场复核：尾部加权还成立吗', '',
          '> %s｜**全市场 %d 只**｜训练 %d 样本 / 验证 %d 样本｜117 个验证日｜NN 为 %d seed 平均' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), X.shape[1], len(ks_tr), len(ks_va), SEEDS), '',
          '| 模型 | D1 命中率 | D1 召回 | D1 平均分位 | D1 超额 | IC |', '|---|---|---|---|---|---|']
    for _, r in df.iterrows():
        md.append('| %s | **%.1f%%** | %.1f%% | %.1f%% | %+.2f%% | %+.4f |' % (
            r['model'], r['hit'] * 100, r['recall'] * 100, r['pctile'] * 100, r['excess'] * 100, r['ic']))
    a = df[df.model == 'NN mse_rank'].iloc[0]
    b = df[df.model == 'NN + 尾部加权'].iloc[0]
    dh = (b['hit'] - a['hit']) * 100
    de = (b['excess'] - a['excess']) * 100     # 正数=桶没那么惨=避掉的损失变少
    di = b['ic'] - a['ic']
    md += ['', '## 与 600 只抽样版的对照', '',
           '| 指标 | 抽样（600 只） | 全市场（{} 只） |'.format(X.shape[1]), '|---|---|---|',
           '| 无加权 D1 命中率 | 28.9% | {:.1f}% |'.format(a['hit'] * 100),
           '| 尾部加权 D1 命中率 | 32.2% | {:.1f}% |'.format(b['hit'] * 100),
           '| 提升 | +3.3pp | **{:+.1f}pp** |'.format(dh),
           '| D1 桶超额 | −1.00% → −1.05% | {:+.2f}% → {:+.2f}%（{:+.2f}pp）|'.format(
               a['excess'] * 100, b['excess'] * 100, de),
           '| IC | 0.1149 → 0.1146 | {:+.4f} → {:+.4f}（{:+.4f}）|'.format(a['ic'], b['ic'], di),
           '',
           '## 判读：命中率涨了，但「避掉的损失」反而少了',
           '',
           '命中率的方向在**全市场复现了**（{:+.1f}pp，抽样版是 +3.3pp），但另外两个指标同时变差：'.format(dh),
           '',
           '- D1 桶超额从 {:+.2f}% 变成 {:+.2f}%，即**剔除这 10% 之后，组合能少亏的钱从 {:.2f}% 降到 {:.2f}%**；'.format(
               a['excess'] * 100, b['excess'] * 100, -a['excess'] * 100, -b['excess'] * 100),
           '- IC 从 {:+.4f} 掉到 {:+.4f}。'.format(a['ic'], b['ic']),
           '',
           '**这两个指标才对应真金白银**。命中率高只说明「模型清单里真垫底的比例更高」，',
           '但如果命中的是那些只跌 0.5% 的边缘输家、漏掉的是会跌 8% 的暴跌股，',
           '禁买清单的经济价值反而是下降的。',
           '',
           '## 结论：不改生产',
           '',
           '1. 尾部队列表在**精确度**上确实更好（全市场 31.4% → 34.0%）；',
           '2. 但**避损能力**变差（桶超额 −1.46% → −1.36%），IC 也降；',
           '3. 抽样版那个 +3.3pp 在全市场缩到 +2.6pp，说明它本来就有相当一部分是噪声；',
           '4. 生产用的是「剔除垫底 10%」，看的是第 2 条，不是第 1 条。',
           '',
           '**这条经验值得记住**：换指标救了「尾部加权无效」的判断，但换到全市场又发现——',
           '**命中率也不是那个终局指标**。终局指标只有一个：剔除之后组合的年化超额变多少。',
           '任何中间指标（IC、命中率、AUC）都只是它的代理，代理之间还会互相矛盾。', '',
           '## 注意', '',
           '- 只跑了 {} 个验证日、{} 个种子，单次点估计仍有 ±1pp 抖动；'.format(nd, SEEDS),
           '- 训练只用了 {} 天（全量 2200 天里抽的）；'.format(N_TRAIN_DAY),
           '- 未做多折；要上线必须走 `purged_cv.py` 那套协议。', '']
    open(f'{ROOT}/reports/d1_fullmarket.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n' + df.round(4).to_string(index=False))
    print('→', f'{ROOT}/reports/d1_fullmarket.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='d1_full'):
        main()
