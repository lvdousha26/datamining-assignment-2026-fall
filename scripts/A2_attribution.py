# -*- coding: utf-8 -*-
"""A2 因子归因：Brinson 分解（行业配置 vs 个股选择）+ 子策略边际贡献

Brinson（单期）：
  配置效应 = Σ(w_p,i − w_b,i) × r_b,i
  选择效应 = Σ w_b,i × (r_p,i − r_b,i)
  交互效应 = Σ(w_p,i − w_b,i) × (r_p,i − r_b,i)
基准 = 全市场等权。因 Y2 已截面去均值，基准总超额≈0，故三分量之和 ≈ 组合超额。
"""
import numpy as np, pandas as pd

ROOT = 'D:/quant'
STEP, HOLD, TOP, COST, CAP = 5, 60, 300, 0.0015, 0.05

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
Y2, M, dates = z['Y2'], z['M'], pd.to_datetime(z['dates'])
T, N = Y2.shape
syms = [str(s).zfill(6) for s in z['symbols']]
l1 = pd.read_parquet(f'{ROOT}/data/big/sw_l1.parquet').drop_duplicates('symbol')
l1map = dict(zip(l1.symbol, l1.l1))
L1 = np.array([l1map.get(s, '') for s in syms])

fp = pd.read_parquet(f'{ROOT}/data/big/fin_panel.parquet')
fp['date'] = pd.to_datetime(fp['date'])
fp = fp[fp.date.isin(set(dates))].drop_duplicates(['date', 'symbol'])
ti = pd.Index(dates).get_indexer(pd.DatetimeIndex(fp.date))
si = pd.Index(syms).get_indexer(fp.symbol.astype(str).str.zfill(6))
ok = (ti >= 0) & (si >= 0)
fp = fp.loc[ok].reset_index(drop=True); ti, si = ti[ok], si[ok]


def cz(v, t):
    return pd.Series(v).groupby(t).transform(lambda x: (x - x.mean()) / (x.std() + 1e-12)).values


QUAL = ['roe_ttm', 'gpa', 'ocf_ni', 'accrual', 'debt_ratio', 'roe_stab']
QDIR = np.array([1, 1, 1, -1, -1, -1])
VAL = ['bp', 'ep', 'sp', 'ocf_yield', 'fcf_yield']
Q = np.zeros((T, N), np.float32); Qc = np.zeros((T, N), np.float32)
for k, c in enumerate(QUAL):
    zz = np.full((T, N), np.nan, np.float32)
    zz[ti, si] = cz(fp[c].values.astype(float), ti) * QDIR[k]
    Q += np.nan_to_num(zz); Qc += np.isfinite(zz)
Q = np.where(Qc >= 3, Q / np.maximum(Qc, 1), np.nan)
Vv = np.zeros((T, N), np.float32); Vc = np.zeros((T, N), np.float32)
for c in VAL:
    zz = np.full((T, N), np.nan, np.float32)
    zz[ti, si] = cz(fp[c].values.astype(float), ti)
    Vv += np.nan_to_num(zz); Vc += np.isfinite(zz)
Vv = np.where(Vc >= 2, Vv / np.maximum(Vc, 1), np.nan)
Q[~M] = np.nan; Vv[~M] = np.nan
eff = np.isfinite(Q) & np.isfinite(Vv) & M


def pick(t, mode, cap=CAP):
    m = eff[t]
    if m.sum() < 500: return None
    idx = np.where(m)[0]
    q, v = Q[t, idx], Vv[t, idx]
    if mode == 'val':
        return idx[np.argsort(-v)[:TOP]]
    if mode == 'qua':
        return idx[np.argsort(-q)[:TOP]]
    pool = idx[q >= np.quantile(q, 0.60)]
    if len(pool) < TOP: return None
    order = pool[np.argsort(-Vv[t, pool])]
    if cap is None: return order[:TOP]
    k = max(int(TOP * cap), 1)
    cnt, sel = {}, []
    for s in order:
        ind = L1[s]
        if ind == '': sel.append(s)
        else:
            if cnt.get(ind, 0) >= k: continue
            cnt[ind] = cnt.get(ind, 0) + 1; sel.append(s)
        if len(sel) >= TOP: break
    return np.array(sel)


PPY = 252 / HOLD
rows = []
for t0 in range(0, T - HOLD - STEP, HOLD):
    hold = pick(t0, 'mix')
    if hold is None or len(hold) < 50: continue
    # 持有期收益（12 个 5 日窗口）
    rets = []
    for k in range(0, HOLD, STEP):
        y = Y2[t0 + k, hold]
        rets.append(np.nanmean(y[np.isfinite(y)]))
    r_p = float(np.prod(1 + np.array(rets)) - 1)

    # Brinson：对持有期内 12 个窗口分别分解再累加
    alloc_t = select_t = inter_t = 0.0
    ind_p = L1[hold]
    for k in range(0, HOLD, STEP):
        tk = t0 + k
        yv = Y2[tk, :]
        mkt = M[tk] & np.isfinite(yv)
        if mkt.sum() < 500: continue
        ind_m = L1[mkt]
        wp = pd.Series(ind_p).value_counts(normalize=True)
        wb = pd.Series(ind_m).value_counts(normalize=True)
        yp = pd.Series(yv[hold]).groupby(ind_p).mean()
        yb_all = pd.Series(yv[mkt]).groupby(ind_m).mean()
        idx = wp.index.union(wb.index)
        wp = wp.reindex(idx).fillna(0); wb = wb.reindex(idx).fillna(0)
        yp = yp.reindex(idx).fillna(0); yb = yb_all.reindex(idx).fillna(0)
        alloc_t += float(((wp - wb) * yb).sum())
        select_t += float((wb * (yp - yb)).sum())
        inter_t += float(((wp - wb) * (yp - yb)).sum())
    alloc, select, inter = alloc_t, select_t, inter_t
    rows.append((t0, r_p, alloc, select, inter, len(hold)))

d = pd.DataFrame(rows, columns=['t', 'r', 'alloc', 'select', 'inter', 'n'])
print('期数 %d | 组合超额累计 %.2f%% | 三分量之和 %.2f%%' %
      (len(d), ((1 + d.r).prod() - 1) * 100,
       ((1 + d.alloc + d.select + d.inter).prod() - 1) * 100), flush=True)


def ann(x):
    return ((1 + x).prod() ** (PPY / len(x)) - 1) * 100


print('\n=== ① Brinson 分解（年化，算术口径）===')
tot = abs(d.alloc.mean()) + abs(d.select.mean()) + abs(d.inter.mean())
print('  组合超额(算术)  %+7.2f%%' % (d.r.mean() * PPY * 100))
print('  行业配置效应    %+7.2f%%  (%.0f%%)' % (d.alloc.mean() * PPY * 100, abs(d.alloc.mean()) / tot * 100))
print('  个股选择效应    %+7.2f%%  (%.0f%%)' % (d.select.mean() * PPY * 100, abs(d.select.mean()) / tot * 100))
print('  交互效应        %+7.2f%%  (%.0f%%)' % (d.inter.mean() * PPY * 100, abs(d.inter.mean()) / tot * 100))
print('  三项之和        %+7.2f%%' % ((d.alloc + d.select + d.inter).mean() * PPY * 100))
print('  （复利年化超额 %+.2f%%，与算术口径的差异来自复利）' % ann(d.r))

print('\n=== ② 子策略边际贡献（年化超额，净）===')
res = {}
for nm, mode, cap in [('纯质量 top300', 'qua', None), ('纯估值 top300', 'val', None),
                      ('质量门槛→估值（无约束）', 'mix', None), ('质量门槛→估值（≤5%行业）', 'mix', CAP)]:
    rr, prev = [], None
    for t0 in range(0, T - HOLD - STEP, HOLD):
        hold = pick(t0, mode, cap)
        if hold is None or len(hold) < 50: continue
        rets = []
        for k in range(0, HOLD, STEP):
            y = Y2[t0 + k, hold]
            rets.append(np.nanmean(y[np.isfinite(y)]))
        r = float(np.prod(1 + np.array(rets)) - 1)
        w = np.zeros(N); w[hold] = 1.0 / len(hold)
        to = 0.0 if prev is None else float(np.abs(w - prev).sum() / 2)
        prev = w
        rr.append(r - to * 2 * COST)
    a = ann(pd.Series(rr)); res[nm] = a
    print('  %-26s %+7.2f%%' % (nm, a))

print('\n=== ③ 分年配置/选择效应 ===')
d['yr'] = pd.DatetimeIndex(dates[d.t]).year
for y, g in d.groupby('yr'):
    print('  %d: 超额 %+6.1f%% | 配置 %+6.1f%% | 选择 %+6.1f%% | 交互 %+6.1f%%' %
          (y, ((1 + g.r).prod() - 1) * 100, ((1 + g.alloc).prod() - 1) * 100,
           ((1 + g.select).prod() - 1) * 100, ((1 + g.inter).prod() - 1) * 100))

d.to_csv(f'{ROOT}/reports/A2_attribution.csv', index=False)
print('\n已存 reports/A2_attribution.csv')
