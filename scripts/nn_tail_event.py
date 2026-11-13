# -*- coding: utf-8 -*-
"""10 日标签 + 尾部加权 + 事件门控：神经网络到底值不值得？

三件事一起做（按用户指定的顺序）：
  ① 先用 LightGBM 把 **10 日标签** 的基线跑出来（同一批抽样、同一套 42 因子）
  ② 神经网络 mse_rank 打底 + **尾部加权** w = 1 + 4·(rank<0.1)
  ③ 事件做成**门控**（业绩预告 / ST 状态 → embedding → 乘到 GRU 隐状态），不拼进序列

事件编码
--------
  evt: 0=近期无预告, 1=利好(预增/略增/扭亏/续盈/减亏), 2=利空(预减/略减/首亏/续亏/增亏), 3=不确定
       取「最近 60 日内」的最近一条
  st : 0/1，当日是否 ST（st_calendar）

用法：python scripts/nn_tail_event.py
产出：reports/nn_tail_event.md + csv
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
L, H = 30, 10
N_SYM, N_DAY = 600, 400
BATCH, EPOCHS = 512, 3
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
POS = {'预增', '略增', '扭亏', '续盈', '减亏'}
NEG = {'预减', '略减', '首亏', '续亏', '增亏'}


def fwd_excess(R, h=H):
    """t 日信号 → t+1 收盘买入 → 持有 h 日"""
    T, N = R.shape
    acc = np.ones_like(R, dtype=np.float32)
    cnt = np.zeros((T, N), np.int8)
    for k in range(2, h + 2):
        sh = np.full_like(R, np.nan)
        sh[:T - k] = R[k:]
        ok = np.isfinite(sh)
        acc = acc * np.where(ok, 1 + sh, 1.0).astype(np.float32)
        cnt += ok
    out = np.where(cnt == h, acc - 1, np.nan)
    return out - np.nanmean(out, 1, keepdims=True)


def build_events(dates, syms, col):
    """返回 evt[T,N]（0..3）与 st[T,N]（0/1）"""
    T, N = len(dates), len(syms)
    di = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    evt = np.zeros((T, N), np.int8)
    st = np.zeros((T, N), np.int8)

    fs = sorted(glob.glob(f'{ROOT}/data/big/yjyg/*.parquet'))
    y = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    y['sym'] = y['股票代码'].astype(str).str.zfill(6)
    y['d'] = pd.to_datetime(y['公告日期'], errors='coerce')
    y = y.dropna(subset=['d'])
    y = y[y['sym'].isin(col)]
    y['code'] = y['预告类型'].map(lambda t: 1 if t in POS else 2 if t in NEG else 3)
    for s, g in y.groupby('sym'):
        j = col[s]
        for _, r in g.sort_values('d').iterrows():
            t = di.get(pd.Timestamp(r['d']))
            if t is None:
                t = int(np.searchsorted(np.asarray(dates.values), np.datetime64(r['d'])))
                if t >= T:
                    continue
            evt[t:min(t + 60, T), j] = r['code']     # 60 日内有效
    print('  预告覆盖：%.1f%% 的 (日,股)' % (100 * (evt > 0).mean()), flush=True)

    sc = pd.read_parquet(f'{ROOT}/data/big/st_calendar.parquet')
    sc['sym'] = sc['symbol'].astype(str).str.zfill(6)
    sc = sc[sc['sym'].isin(col)]
    for s, g in sc.groupby('sym'):
        j = col[s]
        ii = [di[pd.Timestamp(x)] for x in g['date'] if pd.Timestamp(x) in di]
        st[ii, j] = 1
    print('  ST 覆盖：%.2f%% 的 (日,股)' % (100 * st.mean()), flush=True)
    return evt, st


class Net(nn.Module):
    def __init__(self, f, d=64, use_gate=False):
        super().__init__()
        self.gru = nn.GRU(f, d, batch_first=True)
        self.use_gate = use_gate
        if use_gate:
            self.e_evt = nn.Embedding(4, d)
            self.e_st = nn.Embedding(2, d)
            self.g = nn.Linear(d, d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x, evt=None, st=None):
        z, _ = self.gru(x)
        h = z[:, -1]
        if self.use_gate and evt is not None:
            h = h * torch.sigmoid(self.g( self.e_evt(evt) + self.e_st(st) ))
        return self.head(h).squeeze(-1)


def gather(X, y, evt, st, days, idx):
    S, Y, T_, E, Sg = [], [], [], [], []
    for t in days:
        if t < L or not np.isfinite(y[t]).any():
            continue
        sel = idx[np.isfinite(y[t, idx])]
        if len(sel) < 200:
            continue
        S.append(X[t - L + 1:t + 1][:, sel, :].transpose(1, 0, 2).astype(np.float32))
        Y.append(y[t, sel].astype(np.float32))
        E.append(evt[t, sel]); Sg.append(st[t, sel])
        T_ += [t] * len(sel)
    return (np.concatenate(S), np.concatenate(Y), np.array(T_),
            np.concatenate(E), np.concatenate(Sg))


def rank_target(y, tt):
    out = np.zeros(len(y), np.float32)
    for t in np.unique(tt):
        m = tt == t
        v = y[m]
        out[m] = (np.argsort(np.argsort(v)) + 0.5) / len(v)
    return out


def daily_ic(p, y, tt):
    from scipy.stats import spearmanr
    ics = []
    for t in np.unique(tt):
        m = tt == t
        if m.sum() >= 200:
            ics.append(spearmanr(p[m], y[m]).statistic)
    s = pd.Series(ics).dropna()
    return float(s.mean()), float(s.mean() / s.std() if s.std() else np.nan), len(s)


def train_nn(Xtr, ytr, ttr, etr, str_, Xva, yva, tva, eva, sva, gate=False,
             tail=False, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(Xtr.shape[2], use_gate=gate).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    rk = rank_target(ytr, ttr)
    w = np.where(rk < 0.1, 1 + 4.0, 1.0).astype(np.float32) if tail else np.ones(len(rk), np.float32)
    n = len(Xtr); steps = (n + BATCH - 1) // BATCH
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps * epochs)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for b in range(steps):
            ix = perm[b * BATCH:(b + 1) * BATCH]
            xb = torch.from_numpy(Xtr[ix]).to(DEV)
            out = torch.sigmoid(m(xb, torch.from_numpy(etr[ix]).long().to(DEV),
                                  torch.from_numpy(str_[ix]).long().to(DEV)) if gate else m(xb))
            tgt = torch.from_numpy(rk[ix]).to(DEV)
            wb = torch.from_numpy(w[ix]).to(DEV)
            loss = (wb * (out - tgt) ** 2).sum() / wb.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()
    m.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(Xva), 4096):
            xb = torch.from_numpy(Xva[s:s + 4096]).to(DEV)
            if gate:
                z = m(xb, torch.from_numpy(eva[s:s + 4096]).long().to(DEV),
                      torch.from_numpy(sva[s:s + 4096]).long().to(DEV))
            else:
                z = m(xb)
            preds.append(z.cpu().numpy())
    return daily_ic(np.concatenate(preds), yva, tva)


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, M = z['X'], z['M']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    syms = [str(s).zfill(6) for s in z['symbols']]
    col = {s: j for j, s in enumerate(syms)}
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    y = fwd_excess(d['R'])
    print('fwd%d 超额：均值 %.3f%% 标准差 %.2f%%' % (H, np.nanmean(y) * 100, np.nanstd(y) * 100), flush=True)

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

    # ---- ① LGBM 基线（10 日标签）----
    t0 = time.time()
    Ftr = Xtr[:, -1, :]
    Fva = Xva[:, -1, :]
    agg_tr = np.concatenate([Xtr.mean(1), Xtr.std(1)], 1)
    agg_va = np.concatenate([Xva.mean(1), Xva.std(1)], 1)
    rk_tr = rank_target(ytr, ttr)
    for tag, A, B in [('LightGBM（当期因子）', Ftr, Fva),
                      ('LightGBM（+30日均值/波动）', np.concatenate([Ftr, agg_tr], 1),
                       np.concatenate([Fva, agg_va], 1))]:
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               objective='binary', verbose=-1)
        m.fit(A, (rk_tr >= 0.5).astype(int))         # 预测「上半区」，概率与收益同向
        ic, icir, nd = daily_ic(m.predict_proba(B)[:, 1], yva, tva)
        print('  %-24s IC %+.4f | ICIR %+.2f | %.0fs' % (tag, ic, icir, time.time() - t0), flush=True)
        rows.append(dict(model=tag, ic=ic, icir=icir, days=nd, sec=time.time() - t0))

    # ---- ②③ NN 三档 ----
    for tag, gate, tail in [('NN mse_rank', False, False),
                            ('NN + 尾部加权', False, True),
                            ('NN + 尾部加权 + 事件门控', True, True)]:
        t0 = time.time()
        res = [train_nn(Xtr, ytr, ttr, etr, str_, Xva, yva, tva, eva, sva,
                        gate=gate, tail=tail, seed=s) for s in (0, 1, 2)]
        ic = float(np.mean([r[0] for r in res]))
        icir = float(np.mean([r[1] for r in res]))
        print('  %-24s IC %+.4f | ICIR %+.2f | %.0fs' % (tag, ic, icir, time.time() - t0), flush=True)
        rows.append(dict(model=tag, ic=ic, icir=icir, days=res[0][2], sec=time.time() - t0))

    df = pd.DataFrame(rows).sort_values('ic', ascending=False)
    df.to_csv(f'{ROOT}/reports/nn_tail_event.csv', index=False)
    md = ['# 10 日标签：LGBM 基线 vs 尾部加权 NN vs 事件门控', '',
          '> %s｜输入 30×42 量价｜标签 fwd%d 超额（截面去均值）｜训练 %d / 验证 %d｜NN 为 3 seed 均值' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), H, len(Xtr), len(Xva)), '',
          '| 模型 | IC | ICIR | 相对 LGBM |', '|---|---|---|---|']
    b = df[df.model.str.startswith('LightGBM（当期')]['ic'].iloc[0] if (df.model.str.startswith('LightGBM（当期')).any() else df.iloc[-1]['ic']
    for _, r in df.iterrows():
        md.append('| %s | **%+.4f** | %+.2f | %+.4f |' % (r['model'], r['ic'], r['icir'], r['ic'] - b))
    md += ['', '## 怎么读', '',
           '1. **先看 LGBM 那两行**：如果神经网络打不过「当期 42 因子 + 30 日均值」这种平价特征，'
           '那「30 日序列 + 门控」的复杂度就没有支付理由。',
           '2. **尾部加权的效果**看 `NN mse_rank` → `NN + 尾部加权` 的差：'
           '若 IC 上升，说明模型确实把容量花在了有用的那一撮上（IC 是对全截面算的，'
           '尾部样本变准会拉动整体秩相关）。',
           '3. **事件门控**看最后一行与上一行的差。事件覆盖率只有百分之几，'
           '若差异落在种子噪声（约 ±0.005）以内，说明门控没起作用——这本身是个结论：'
           '稀疏事件在日频选股里信息量太低。', '',
           '## 局限', '',
           '- 600 只 × 400 日抽样，样本比全量小两个量级，绝对值不可与主模型 0.1041 直接比；',
           '- 事件编码只用「最近 60 日有无/方向」，没有用金额、行业、连板等细节；',
           '- 10 日标签下 purged CV 的 embargo 应是 10 而非 6，这里只是单段训练/验证，未做多折。', '']
    open(f'{ROOT}/reports/nn_tail_event.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n' + df.round(4).to_string(index=False))
    print('→', f'{ROOT}/reports/nn_tail_event.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='nn_tail_event'):
        main()
