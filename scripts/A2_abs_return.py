# -*- coding: utf-8 -*-
"""重算 A2 的绝对收益（不再用「超额 + 指数」近似）

问题：Y2 是「截面去均值」的超额，基准是全市场等权；之前用沪深300 近似市场，
      在 2024-2026 小盘暴涨期会严重失真。
做法：直接算 A2 持仓的原始 5 日收益（停在 close>0 且 |fwd5|<50% 的样本上），
      12 个窗口复利 → 真实绝对收益，与 A1 同口径复合。
"""
import numpy as np, pandas as pd, glob, io, contextlib, os, sys

ROOT, BASE = 'D:/quant', r'C:/Users/25097/WorkBuddy/2026-10-30-23-31-35'
STEP, HOLD, TOP, COST, CAP = 5, 60, 300, 0.0015, 0.05

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
Y2, M, dates = z['Y2'], z['M'], pd.to_datetime(z['dates'])
T, N = Y2.shape
syms = [str(s).zfill(6) for s in z['symbols']]

# ---- 原始 5 日收益（严格清洗）----
px = []
for f in sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet')):
    d = pd.read_parquet(f, columns=['date', 'symbol', 'close'])
    d['symbol'] = d['symbol'].astype(str).str.zfill(6)
    px.append(d)
px = pd.concat(px, ignore_index=True).sort_values(['symbol', 'date'])
px.loc[px['close'] <= 0, 'close'] = np.nan
px['f5'] = px.groupby('symbol')['close'].transform(lambda s: s.shift(-5) / s - 1)
px['f5'] = px['f5'].replace([np.inf, -np.inf], np.nan).clip(-1.0, 3.0)
F = np.full((T, N), np.nan, np.float32)
ti = pd.Index(dates).get_indexer(pd.DatetimeIndex(px['date']))
si = pd.Index(syms).get_indexer(px['symbol'])
ok = (ti >= 0) & (si >= 0) & np.isfinite(px['f5'].values)
F[ti[ok], si[ok]] = px['f5'].values[ok]
print('原始收益矩阵 非空 %.1f%% | 中位 %.4f | p5 %.4f | p95 %.4f' %
      (np.isfinite(F).mean() * 100, np.nanmedian(F), np.nanpercentile(F, 5), np.nanpercentile(F, 95)), flush=True)

# ---- 因子 ----
l1 = pd.read_parquet(f'{ROOT}/data/big/sw_l1.parquet').drop_duplicates('symbol')
L1 = np.array([dict(zip(l1.symbol, l1.l1)).get(s, '') for s in syms])
fp = pd.read_parquet(f'{ROOT}/data/big/fin_panel.parquet')
fp['date'] = pd.to_datetime(fp['date'])
fp = fp[fp.date.isin(set(dates))].drop_duplicates(['date', 'symbol'])
ti2 = pd.Index(dates).get_indexer(pd.DatetimeIndex(fp.date))
si2 = pd.Index(syms).get_indexer(fp.symbol.astype(str).str.zfill(6))
ok2 = (ti2 >= 0) & (si2 >= 0)
fp = fp.loc[ok2].reset_index(drop=True); ti2, si2 = ti2[ok2], si2[ok2]


def cz(v, t):
    return pd.Series(v).groupby(t).transform(lambda x: (x - x.mean()) / (x.std() + 1e-12)).values


QUAL = ['roe_ttm', 'gpa', 'ocf_ni', 'accrual', 'debt_ratio', 'roe_stab']
QDIR = np.array([1, 1, 1, -1, -1, -1])
VAL = ['bp', 'ep', 'sp', 'ocf_yield', 'fcf_yield']
Q = np.zeros((T, N), np.float32); Qc = np.zeros((T, N), np.float32)
for k, c in enumerate(QUAL):
    zz = np.full((T, N), np.nan, np.float32)
    zz[ti2, si2] = cz(fp[c].values.astype(float), ti2) * QDIR[k]
    Q += np.nan_to_num(zz); Qc += np.isfinite(zz)
Q = np.where(Qc >= 3, Q / np.maximum(Qc, 1), np.nan)
Vv = np.zeros((T, N), np.float32); Vc = np.zeros((T, N), np.float32)
for c in VAL:
    zz = np.full((T, N), np.nan, np.float32)
    zz[ti2, si2] = cz(fp[c].values.astype(float), ti2)
    Vv += np.nan_to_num(zz); Vc += np.isfinite(zz)
Vv = np.where(Vc >= 2, Vv / np.maximum(Vc, 1), np.nan)
Q[~M] = np.nan; Vv[~M] = np.nan
eff = np.isfinite(Q) & np.isfinite(Vv) & M

# ---- 回测（绝对收益）----
rows, prev = [], None
for t0 in range(0, T - HOLD - STEP, HOLD):
    m = eff[t0]
    if m.sum() < 500: continue
    idx = np.where(m)[0]
    q = Q[t0, idx]
    pool = idx[q >= np.quantile(q, 0.60)]
    if len(pool) < TOP: continue
    order = pool[np.argsort(-Vv[t0, pool])]
    k = max(int(TOP * CAP), 1)
    cnt, sel = {}, []
    for s in order:
        ind = L1[s]
        if ind == '': sel.append(s)
        else:
            if cnt.get(ind, 0) >= k: continue
            cnt[ind] = cnt.get(ind, 0) + 1; sel.append(s)
        if len(sel) >= TOP: break
    hold = np.array(sel)
    rets, mkt = [], []
    for kk in range(0, HOLD, STEP):
        y = F[t0 + kk, hold]
        rets.append(np.nanmean(y[np.isfinite(y)]))
        mm = np.isfinite(F[t0 + kk])      # 市场用全部有收益的样本，不经过 M 掩码
        mkt.append(np.nanmean(F[t0 + kk][mm]))
    r = float(np.prod(1 + np.array(rets)) - 1)
    rm = float(np.prod(1 + np.array(mkt)) - 1)
    w = np.zeros(N); w[hold] = 1.0 / len(hold)
    to = 0.0 if prev is None else float(np.abs(w - prev).sum() / 2)
    prev = w
    rows.append((t0, r, rm, to))
d = pd.DataFrame(rows, columns=['t', 'r', 'mkt', 'to'])
d['net'] = d['r'] - d['to'] * 2 * COST
d['ex'] = d['net'] - d['mkt']
PPY = 252 / HOLD


def ann(x):
    return ((1 + x).prod() ** (PPY / len(x)) - 1) * 100


print('\n=== A2 真实绝对收益（%d 期）===' % len(d))
print('  绝对            年化 %+7.2f%% 波动 %6.2f%%' % (ann(d.net), d.net.std() * np.sqrt(PPY) * 100))
print('  全市场等权      年化 %+7.2f%% 波动 %6.2f%%' % (ann(d.mkt), d.mkt.std() * np.sqrt(PPY) * 100))
print('  超额（相对等权）年化 %+7.2f%% 波动 %6.2f%%' % (ann(d.ex), d.ex.std() * np.sqrt(PPY) * 100))
print('  → 之前用「超额+沪深300」得到 13.39%%，实际 %.2f%%' % ann(d.net))

print('\n=== 分年 ===')
d['yr'] = pd.DatetimeIndex(dates[d.t]).year
for y, g in d.groupby('yr'):
    print('  %d: 绝对 %+7.1f%% | 市场 %+7.1f%% | 超额 %+6.1f%%' % (y, ann(g.net), ann(g.mkt), ann(g.ex)))

# 月度化（季度拆 3 个月）
rec = []
for _, r in d.iterrows():
    st = dates[int(r.t)].to_period('M')
    for j in range(3):
        rec.append((st + j + 1, (1 + r.net) ** (1 / 3) - 1))
a2m = pd.DataFrame(rec, columns=['m', 'r']).groupby('m').r.apply(lambda x: (1 + x).prod() - 1)
a2m.index = pd.PeriodIndex(a2m.index, freq='M')
a2m.to_pickle(f'{ROOT}/data/big/a2_abs_monthly.pkl')
print('\n已存 data/big/a2_abs_monthly.pkl（月度绝对收益 %d 期）' % len(a2m))
