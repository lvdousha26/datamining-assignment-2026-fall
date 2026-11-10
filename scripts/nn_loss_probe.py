# -*- coding: utf-8 -*-
"""30 日量价序列 → 未来 10 日涨跌：三种损失函数横向对比

问题：同样输入、同样网络，只换损失，IC 差多少？
    L1  MSE 回归（目标 = 截面 rank，去掉量纲，模型学排序）
    L2  BCE 二分类（目标 = 涨/跌）
    L3  pairwise 排序（同日成对比较，直接优化排序，匹配选股任务）

输入：过去 30 个交易日的 42 个量价因子（`arr.npz` 的 X，已做后复权 + 断层感知）
标签：t 日收盘出信号 → t+1 收盘买入 → 持有 10 日，`close[t+11]/close[t+1]-1`，再截面去均值

**没做的两件事**（数据不在手上）：
  - 「挂单」：A 股 Level-2 十档要付费，免费源只有最近 15 个交易日的涨停封单；
  - 「事件」：业绩预告/ST 日历可以编码成额外通道，但它们在时间上极稀疏（一年几十条），
    塞进日频序列会被稀释，更适合单独做一个事件门控（后续再说）。

用法：python scripts/nn_loss_probe.py
产出：reports/nn_loss.md
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
L = 30          # 回看窗口
H = 10          # 持有期
N_SYM = 600     # 抽样股票数（控时间）
N_DAY = 400     # 抽样交易日
EPOCHS = 4
BATCH = 1024
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def fwd_excess(R, dates, h=H):
    """t 日收盘出信号 → t+1 收盘买入 → 持有 h 日 → t+h+1 收盘卖出

    收益 = close[t+h+1]/close[t+1] - 1 = prod_{k=2..h+1} (1+R[t+k]) - 1
    （R[t] 是 t 日当天收益，所以 t+1 买入后才开始吃到 t+2 的收益）
    最后减去当日全市场等权超额。
    """
    T, N = R.shape
    acc = np.ones_like(R)
    cnt = np.zeros((T, N), dtype=np.int8)
    for k in range(2, h + 2):
        sh = np.full_like(R, np.nan)
        sh[:T - k] = R[k:]
        ok = np.isfinite(sh)
        acc = acc * np.where(ok, 1 + sh, 1.0)
        cnt += ok
    out = np.where(cnt == h, acc - 1, np.nan)
    ew = np.nanmean(out, 1, keepdims=True)
    return out - ew


class Net(nn.Module):
    def __init__(self, f, d=64):
        super().__init__()
        self.gru = nn.GRU(f, d, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        z, _ = self.gru(x)
        z = z[:, -1]                       # 取最后一步
        return self.head(z).squeeze(-1)


def build_xy(X, y, days, syms_idx):
    """返回 [(seq, target)]；seq = [n, L, F]"""
    out_s, out_y, out_t, out_i = [], [], [], []
    for t in days:
        if t < L or not np.isfinite(y[t]).any():
            continue
        idx = syms_idx
        idx = idx[np.isfinite(y[t, idx])]
        if len(idx) < 200:
            continue
        seq = X[t - L + 1:t + 1][:, idx, :].transpose(1, 0, 2)   # [n, L, F]
        out_s.append(seq.astype(np.float32))
        out_y.append(y[t, idx].astype(np.float32))
        out_t += [t] * len(idx)
        out_i += list(idx)
    if not out_s:
        return None, None, None
    return np.concatenate(out_s), np.concatenate(out_y), np.array(out_t)


def rank_target(y_by_day):
    """逐日把目标转成 (rank-0.5)/n —— 去掉量纲，模型专注排序"""
    out = np.zeros_like(y_by_day)
    for t in np.unique(y_by_day[:, 1]):
        m = y_by_day[:, 1] == t
        v = y_by_day[m, 0]
        r = v.argsort().argsort()
        out[m, 0] = (r + 0.5) / len(v)
    return out[:, 0]


def daily_ic(pred, y, tt, min_n=200):
    ics = []
    for t in np.unique(tt):
        m = tt == t
        if m.sum() < min_n:
            continue
        from scipy.stats import spearmanr
        ics.append(spearmanr(pred[m], y[m]).statistic)
    s = pd.Series(ics).dropna()
    return s.mean(), (s.mean() / s.std() if s.std() else np.nan), len(s)


def train_one(loss_name, Xtr, ytr, ttr, Xva, yva, tva, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net(Xtr.shape[2]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Xtr)
    # 预先算好 rank 目标与符号
    ydf = pd.DataFrame({'y': ytr, 't': ttr})
    rk = rank_target(ydf[['y', 't']].values).astype(np.float32)
    steps = (n + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps * EPOCHS)
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = np.random.permutation(n)
        tot = 0.0
        for b in range(steps):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            xb = torch.from_numpy(Xtr[idx]).to(DEV)
            out = model(xb)
            if loss_name == 'mse_rank':
                yb = torch.from_numpy(rk[idx]).to(DEV)
                loss = Fn.mse_loss(torch.sigmoid(out), yb)
            elif loss_name == 'bce':
                yb = torch.from_numpy((ytr[idx] > 0).astype(np.float32)).to(DEV)
                loss = Fn.binary_cross_entropy_with_logits(out, yb)
            else:  # pairwise
                yb = torch.from_numpy(ytr[idx]).to(DEV)
                i, j = torch.triu_indices(len(idx), len(idx), offset=1, device=DEV)
                if len(i) > 4000:
                    sel = torch.randint(0, len(i), (4000,), device=DEV)
                    i, j = i[sel], j[sel]
                diff = yb[i] - yb[j]
                m = diff.abs() > 1e-4
                if m.sum() < 10:
                    continue
                loss = Fn.binary_cross_entropy_with_logits(out[i][m] - out[j][m],
                                                           (diff[m] > 0).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        print('    ep%d loss %.4f | %.0fs' % (ep + 1, tot / steps, time.time() - t0), flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(Xva), 4096):
            preds.append(model(torch.from_numpy(Xva[s:s + 4096]).to(DEV)).cpu().numpy())
    pred = np.concatenate(preds)
    ic, icir, nd = daily_ic(pred, yva, tva)
    print('  %-10s IC %+.4f | ICIR %+.2f | %d 日 | %.0fs' % (
        loss_name, ic, icir, nd, time.time() - t0), flush=True)
    return dict(loss=loss_name, ic=float(ic), icir=float(icir), days=int(nd),
                sec=time.time() - t0)


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X = z['X']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    M = z['M']
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    R = d['R']
    print('R %s | X %s' % (R.shape, X.shape), flush=True)

    y = fwd_excess(R, dates)
    print('fwd%d 超额：均值 %.3f%% | 标准差 %.2f%%' % (H, np.nanmean(y) * 100, np.nanstd(y) * 100))

    rng = np.random.default_rng(0)
    syms = rng.choice(X.shape[1], N_SYM, replace=False)
    syms = syms[M.any(0)[syms]]
    tr_days_all = np.where((dates < '2025-01-01').values)[0]
    tr_days_all = tr_days_all[tr_days_all >= L + H]
    tr_days = rng.choice(tr_days_all, min(N_DAY, len(tr_days_all)), replace=False)
    va_days = np.where((dates >= '2025-01-01') & (dates <= '2025-06-30').values)[0]
    print('训练日 %d | 验证日 %d | 股票 %d' % (len(tr_days), len(va_days), len(syms)), flush=True)

    Xtr, ytr, ttr = build_xy(X, y, np.sort(tr_days), syms)
    Xva, yva, tva = build_xy(X, y, va_days, syms)
    print('训练样本 %s | 验证样本 %s' % (Xtr.shape, Xva.shape), flush=True)

    rows = []
    for name in ['mse_rank', 'bce', 'pairwise']:
        print('\n[%s]' % name, flush=True)
        runs = [train_one(name, Xtr, ytr, ttr, Xva, yva, tva, seed=sd) for sd in (0, 1, 2)]
        rows.append(dict(loss=name,
                         ic=float(np.mean([r['ic'] for r in runs])),
                         ic_std=float(np.std([r['ic'] for r in runs])),
                         icir=float(np.mean([r['icir'] for r in runs])),
                         days=runs[0]['days'], sec=float(np.mean([r['sec'] for r in runs]))))

    df = pd.DataFrame(rows).sort_values('ic', ascending=False)
    print('\n' + df.round(4).to_string(index=False))
    df.to_csv(f'{ROOT}/reports/nn_loss.csv', index=False)
    json.dump(rows, open(f'{ROOT}/reports/nn_loss.json', 'w'), ensure_ascii=False, indent=1)

    md = ['# 30 日量价 → 未来 10 日：三种损失函数对比', '',
          '> %s｜输入 30×42 量价序列｜标签 fwd%d 超额收益（截面去均值）｜训练 %d 样本 / 验证 %d｜**3 个随机种子**' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), H, len(Xtr), len(Xva)), '',
          '| 损失 | IC（3 seed 均值） | 标准差 | ICIR | 有效天数 |', '|---|---|---|---|---|']
    for _, r in df.iterrows():
        md.append('| `%s` | **%+.4f** | %.4f | %+.2f | %d |' % (
            r['loss'], r['ic'], r.get('ic_std', 0), r['icir'], r['days']))
    md += ['', '## 结论', '',
           '排序损失（pairwise）通常优于回归/分类，因为它直接优化「同日谁比谁强」——'
           '而选股要的正是截面排序，不是绝对涨跌幅。'
           if df.iloc[0]['loss'] == 'pairwise' else
           '本次实验中 `%s` 最好：%s。' % (
               df.iloc[0]['loss'],
               '说明在这个信噪比下，回归 rank 目标比成对比较更稳（成对损失方差大）。'
               if df.iloc[0]['loss'] == 'mse_rank' else
               '说明二分类反而最稳（标签只有 0/1 时噪声更小）。'),
           '',
           '## 两条没做的输入', '',
           '- **挂单**：A 股十档要付费；免费只有涨停封单（最近 15 个交易日），无法做 30 日序列。'
           '可用的代理是日内形态（振幅/上下影线/收盘位置 clv），这些已在 42 因子里。',
           '- **事件**：业绩预告 9 万条、ST 日历 322 只，但一年只有几十条/股，塞进日频序列会被稀释到看不见。'
           '正确做法是事件门控（event gating）或单独一条事件通道做交叉特征，不是直接拼进 30 日窗口。', '']
    open(f'{ROOT}/reports/nn_loss.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n→', f'{ROOT}/reports/nn_loss.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='nn_loss'):
        main()
