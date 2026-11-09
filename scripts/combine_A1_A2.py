# -*- coding: utf-8 -*-
"""A1（策略③）× A2（质量+估值）能否复合

A1：策略③ 六腿，月频，真实收益（人民币口径）
A2：季度调仓，回测输出的是超额（Y2 已截面去均值）→ 绝对收益 = 超额 + 市场
    市场用沪深300 月度收益近似（A2 偏大盘价值，beta≈1）

季度收益按几何拆成 3 个月，与 A1 月度序列对齐。
"""
import io, contextlib, os, sys
import numpy as np, pandas as pd

ROOT = 'D:/quant'
BASE = r'C:/Users/25097/WorkBuddy/2026-10-30-23-31-35'

# ---------- A2 季度收益 ----------
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

HOLD, STEP, TOP, COST, CAP = 60, 5, 300, 0.0015, 0.05
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
        if ind == '':
            sel.append(s)
        else:
            if cnt.get(ind, 0) >= k: continue
            cnt[ind] = cnt.get(ind, 0) + 1; sel.append(s)
        if len(sel) >= TOP: break
    hold = np.array(sel)
    rets = []
    for k2 in range(0, HOLD, STEP):
        y = Y2[t0 + k2, hold]
        rets.append(np.nanmean(y[np.isfinite(y)]))
    r = float(np.prod(1 + np.array(rets)) - 1)
    w = np.zeros(N); w[hold] = 1.0 / len(hold)
    to = 0.0 if prev is None else float(np.abs(w - prev).sum() / 2)
    rows.append((t0, r, to)); prev = w
a2 = pd.DataFrame(rows, columns=['t', 'r', 'to'])
a2['net'] = a2['r'] - a2['to'] * 2 * COST
print('A2 %d 期 | %s ~ %s' % (len(a2), dates[a2.t.min()].date(), dates[a2.t.max()].date()), flush=True)

# ---------- 市场（沪深300 月度） ----------
def sina_abs(f):
    import json
    d = json.load(open(os.path.join(BASE, f)))
    t = pd.DataFrame([(x['day'], float(x['close'])) for x in d], columns=['d', 'c'])
    return t.drop_duplicates('d').set_index(pd.to_datetime(t['d'])).sort_index()['c']


hs = sina_abs('sina_000300.json')
hs_m = (1 + hs.pct_change()).resample('ME').prod() - 1
hs_m.index = hs_m.index.to_period('M')

# A2：季度收益 → 3 个月（几何拆分），起始月 = 调仓月的下一个月
rec = []
for _, r in a2.iterrows():
    t0 = int(r.t)
    start = dates[t0].to_period('M')
    mret = (1 + r.net) ** (1 / 3) - 1
    for j in range(3):
        rec.append((start + j + 1, mret))
a2m = pd.DataFrame(rec, columns=['m', 'ex']).groupby('m').ex.apply(lambda x: (1 + x).prod() - 1)
a2m.index = pd.PeriodIndex(a2m.index, freq='M')
a2_abs = (1 + a2m) * (1 + hs_m.reindex(a2m.index).fillna(0)) - 1

# ---------- A1 ----------
os.chdir(BASE)
with contextlib.redirect_stdout(io.StringIO()):
    sys.path.insert(0, BASE)
    import spec3
os.chdir(ROOT)
a1 = spec3.s3B.dropna()
print('A1 %d 月 | %s ~ %s' % (len(a1), a1.index[0], a1.index[-1]), flush=True)

j = pd.concat([a1.rename('A1'), a2_abs.rename('A2')], axis=1).dropna()
print('对齐 %d 个月 | 相关 %.2f' % (len(j), j.A1.corr(j.A2)), flush=True)


def st(x, nm):
    a = (1 + x).prod() ** (12 / len(x)) - 1
    v = x.std() * 12 ** .5
    nav = (1 + x).cumprod(); dd = (nav / nav.cummax() - 1).min()
    print('  %-22s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%%' %
          (nm, a * 100, v * 100, (a - .02) / v, dd * 100))
    return (a - .02) / v


print('\n=== 权重扫描（月度再平衡）===')
print('%-10s %9s %9s %8s %9s' % ('w_A2', '年化', '波动', '夏普', '回撤'))
best = (None, -9)
for w in [0, .1, .2, .3, .4, .5, .6, .7, 1.0]:
    c = (1 - w) * j.A1 + w * j.A2
    a = (1 + c).prod() ** (12 / len(c)) - 1; v = c.std() * 12 ** .5
    nav = (1 + c).cumprod(); dd = (nav / nav.cummax() - 1).min()
    sh = (a - .02) / v
    print('%-10.0f%% %+8.2f%% %8.2f%% %8.2f %8.1f%%' % (w * 100, a * 100, v * 100, sh, dd * 100))
    if sh > best[1]: best = (w, sh)
print('→ 夏普最优 w_A2 = %.0f%%（%.2f）' % (best[0] * 100, best[1]))

print('\n=== 单独对照 ===')
st(j.A1, 'A1 策略③')
st(j.A2, 'A2 质量+估值')
print('  月收益相关 %.3f | A2 相对 A1 的 beta %.2f' %
      (j.A1.corr(j.A2), np.polyfit(j.A1, j.A2, 1)[0]))
