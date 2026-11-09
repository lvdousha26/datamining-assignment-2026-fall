# -*- coding: utf-8 -*-
"""A2 加行业约束：每申万一级 ≤ cap

配额法：选出 300 只后，按行业计数，超配行业只保留估值分最高的 k = TOP×cap 只，
名额让给下一个行业。对比 cap = 无 / 15% / 5% / 3%。
"""
import numpy as np, pandas as pd

ROOT = 'D:/quant'
STEP, HOLD, TOP, COST = 5, 60, 300, 0.0015

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
Y2, M, dates = z['Y2'], z['M'], pd.to_datetime(z['dates'])
T, N = Y2.shape
syms = [str(s).zfill(6) for s in z['symbols']]

l1 = pd.read_parquet(f'{ROOT}/data/big/sw_l1.parquet').drop_duplicates('symbol')
l1map = dict(zip(l1.symbol, l1.l1))
L1 = np.array([l1map.get(s, '') for s in syms])
print('行业映射覆盖 %d / %d 只 | %d 个一级行业' %
      ((L1 != '').sum(), N, len(set(L1[L1 != '']))), flush=True)

fp = pd.read_parquet(f'{ROOT}/data/big/fin_panel.parquet')
fp['date'] = pd.to_datetime(fp['date'])
fp = fp[fp.date.isin(set(dates))].drop_duplicates(['date', 'symbol'])
ti = pd.Index(dates).get_indexer(pd.DatetimeIndex(fp.date))
si = pd.Index(syms).get_indexer(fp.symbol.astype(str).str.zfill(6))
ok = (ti >= 0) & (si >= 0)
fp = fp.loc[ok].reset_index(drop=True)      # 必须与 ti/si 同步过滤
ti, si = ti[ok], si[ok]


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
print('有效格子 %d' % eff.sum(), flush=True)


def pick(t, cap):
    m = eff[t]
    if m.sum() < 500: return None
    idx = np.where(m)[0]
    q = Q[t, idx]
    pool = idx[q >= np.quantile(q, 0.60)]
    if len(pool) < TOP: return None
    order = pool[np.argsort(-Vv[t, pool])]
    if cap is None:
        return order[:TOP]
    k = max(int(TOP * cap), 1)
    cnt, sel = {}, []
    for s in order:
        ind = L1[s]
        if ind == '':            # 无行业：不占配额，但也不优先
            if len(sel) < TOP: sel.append(s)
            continue
        if cnt.get(ind, 0) >= k: continue
        sel.append(s); cnt[ind] = cnt.get(ind, 0) + 1
        if len(sel) >= TOP: break
    # 若因行业限制没选满，按原顺序补齐（放宽行业约束）
    if len(sel) < TOP:
        s2 = set(sel)
        for s in order:
            if s not in s2:
                sel.append(s)
                if len(sel) >= TOP: break
    return np.array(sel)


def backtest(cap):
    rows, prev = [], None
    for t0 in range(0, T - HOLD - STEP, HOLD):
        hold = pick(t0, cap)
        if hold is None or len(hold) < 50: continue
        rets = []
        for k in range(0, HOLD, STEP):
            y = Y2[t0 + k, hold]
            rets.append(np.nanmean(y[np.isfinite(y)]))
        r = float(np.prod(1 + np.array(rets)) - 1)
        w = np.zeros(N); w[hold] = 1.0 / len(hold)
        to = 0.0 if prev is None else float(np.abs(w - prev).sum() / 2)
        # 行业集中度
        ww = pd.Series(L1[hold]).value_counts(normalize=True)
        hhi = float((ww ** 2).sum())
        mx = float(ww.max())
        rows.append((t0, r, to, len(hold), hhi, mx))
        prev = w
    d = pd.DataFrame(rows, columns=['t', 'r', 'to', 'n', 'hhi', 'maxw']).dropna(subset=['r'])
    d['net'] = d['r'] - d['to'] * 2 * COST
    return d


print('\n%-18s %9s %9s %7s %9s %8s %8s %8s' %
      ('行业约束', '年化', '波动', '夏普', '最大回撤', '换手/年', 'HHI', '最大行业'))
res = {}
for nm, cap in [('无约束', None), ('每级≤15%', 0.15), ('每级≤10%', 0.10),
                ('每级≤5%', 0.05), ('每级≤3%', 0.03)]:
    d = backtest(cap)
    if len(d) < 5: print('%-18s 样本不足' % nm); continue
    ppy = 252 / HOLD
    ann = (1 + d.net).prod() ** (ppy / len(d)) - 1
    vol = d.net.std() * np.sqrt(ppy)
    cum = (1 + d.net).cumprod(); dd = (cum / cum.cummax() - 1).min()
    print('%-18s %+8.2f%% %8.2f%% %7.2f %8.1f%% %8.2f %8.3f %7.1f%%' %
          (nm, ann * 100, vol * 100, ann / vol, dd * 100, d.to.mean() * ppy * 2,
           d.hhi.mean(), d.maxw.mean() * 100))
    res[nm] = d

print('\n=== 分年（净收益 %）===')
tab = pd.DataFrame({k: v.groupby(pd.DatetimeIndex(dates[v.t]).year).apply(
    lambda g: ((1 + g.net).prod() - 1) * 100, include_groups=False) for k, v in res.items()})
print(tab.round(1).to_string())
tab.round(2).to_csv(f'{ROOT}/reports/long_strategy_l1_byyear.csv')
print('\n已存 reports/long_strategy_l1_byyear.csv')
