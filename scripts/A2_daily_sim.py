# -*- coding: utf-8 -*-
"""A2 绝对收益：逐日持仓模拟（v2，修正时序 + 同频基准）

修正三处：
  ① 时序：t 日收盘出信号 → t+1 建仓 → 从 t+1 起才吃收益（旧版在决策当日就吃）
  ② 年化：逐日复利后按自然年统计，不再用「单期 60 日收益 ^4.2」
  ③ 基准：给出三种等权口径 —— 日再平衡 / 60 日同频 / 指数，避免「再平衡红利」被误算成 alpha
"""
import numpy as np, pandas as pd

ROOT = 'D:/quant'
HOLD, TOP, CAP, COST = 60, 300, 0.05, 0.0015

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
Y2, M, dates = z['Y2'], z['M'], pd.to_datetime(z['dates'])
T, N = Y2.shape
syms = [str(s).zfill(6) for s in z['symbols']]

# 上市天数掩码：次新股（前 60 个交易日）不参与选股，也不进基准 —— 否则 2016/2017 的
# 新股连板会把「市场」抬到 +36%，而 A2 因需要财报数据根本选不到它们，超额被系统性低估
dr = np.load(f'{ROOT}/data/big/daily_ret.npz')
R = dr['R'].astype(np.float32)
Rf = np.nan_to_num(R, nan=0.0)
has = np.isfinite(R)
_first = np.full(N, -1)
for _i in range(N):
    _idx = np.where(has[:, _i])[0]
    if len(_idx): _first[_i] = _idx[0]
AGE = np.arange(T)[:, None] - _first[None, :]
OLD = AGE >= 60
IDX = pd.read_pickle(f'{ROOT}/data/big/idx_daily.pkl')

l1 = pd.read_parquet(f'{ROOT}/data/big/sw_l1.parquet').drop_duplicates('symbol')
L1m = dict(zip(l1.symbol, l1.l1))
L1 = np.array([L1m.get(s, '') for s in syms])

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
V = np.zeros((T, N), np.float32); Vc = np.zeros((T, N), np.float32)
for c in VAL:
    zz = np.full((T, N), np.nan, np.float32)
    zz[ti2, si2] = cz(fp[c].values.astype(float), ti2)
    V += np.nan_to_num(zz); Vc += np.isfinite(zz)
V = np.where(Vc >= 2, V / np.maximum(Vc, 1), np.nan)
Q[~M] = np.nan; V[~M] = np.nan
eff = np.isfinite(Q) & np.isfinite(V) & M


def pick(t0):
    m = eff[t0] & has[t0] & OLD[t0]
    if m.sum() < 500: return None
    idx = np.where(m)[0]
    q = Q[t0, idx]
    pool = idx[q >= np.quantile(q, 0.60)]
    if len(pool) < TOP: return None
    order = pool[np.argsort(-V[t0, pool])]
    k = max(int(TOP * CAP), 1)
    cnt, sel = {}, []
    for s in order:
        ind = L1[s]
        if ind == '':
            sel.append(s)
        elif cnt.get(ind, 0) < k:
            cnt[ind] = cnt.get(ind, 0) + 1; sel.append(s)
        if len(sel) >= TOP: break
    wn = np.zeros(N); wn[np.array(sel)] = 1.0 / len(sel)
    return wn


def simulate(pick_fn, rebal_days):
    """t 日收盘出信号 → t+1 收盘建仓并付成本 → 从 t+2 起吃收益（与 Y2 口径一致）"""
    nav = np.ones(T); w = np.zeros(N); turns = []; pending = None
    for t in range(T - 1):
        rp = float(w @ Rf[t])
        nav[t + 1] = nav[t] * (1 + rp)
        if w.sum() > 0:
            w = w * (1 + Rf[t]); w /= w.sum()
        if pending is not None:                 # 昨天收盘的决策，今天收盘执行
            to = np.abs(pending - w).sum() / 2
            if to > 0:
                turns.append((t, to))
                nav[t + 1] *= (1 - to * 2 * COST)
            w = pending; pending = None
        if t in rebal_days:
            wn = pick_fn(t)
            if wn is not None: pending = wn
    return nav, turns


rebal = set(range(0, T - 1, HOLD))

# A2
nav_a, turns = simulate(pick, rebal)
# 基准①：60 日同频等权（同一调仓日，全市场可交易股票等权）
def ew60_pick(t):
    m = has[t] & OLD[t]
    idx = np.where(m)[0]
    if len(idx) < 100: return None
    wn = np.zeros(N); wn[idx] = 1.0 / len(idx)
    return wn
nav_e60, _ = simulate(ew60_pick, rebal)
# 基准②：日再平衡等权
ew_r = np.nanmean(R, axis=1)
nav_ed = np.ones(T); nav_ed[1:] = np.cumprod(1 + np.nan_to_num(ew_r[:-1]))

port = pd.Series(nav_a, index=dates)
e60 = pd.Series(nav_e60, index=dates)
ed = pd.Series(nav_ed, index=dates)
PPY = 244


def stat(s, name):
    a = (1 + s).prod() ** (PPY / len(s)) - 1
    v = s.std() * PPY ** .5
    nv = (1 + s).cumprod(); dd = (nv / nv.cummax() - 1).min()
    print('  %-16s 年化 %+7.2f%% | 波动 %6.2f%% | 夏普 %5.2f | 回撤 %6.1f%%'
          % (name, a * 100, v * 100, (a - 0.02) / v, dd * 100))
    return a


pr, e6r, edr = port.pct_change().fillna(0), e60.pct_change().fillna(0), ed.pct_change().fillna(0)
print('=== A2 逐日模拟（%s ~ %s，%d 日，调仓 %d 次）==='
      % (dates[0].date(), dates[-1].date(), T, len(turns)))
print('  平均单边换手 %.1f%% | 年化成本 %.2f%%'
      % (np.mean([t for _, t in turns]) * 100, sum(t for _, t in turns) * 2 * COST / (T / PPY) * 100))
a_p = stat(pr, 'A2 绝对')
a_6 = stat(e6r, '等权(60日同频)')
a_d = stat(edr, '等权(日再平衡)')
for nm, s in IDX.items():
    r = s.pct_change().reindex(dates).fillna(0)
    stat(r, nm)

print('\n  年化差：A2 - 等权60日 %+.2f%% | A2 - 等权日频 %+.2f%%' % ((a_p - a_6) * 100, (a_p - a_d) * 100))
print('  日收益相关：A2 vs 60日均权 %.3f | A2 vs 300 %.3f' %
      (pr.corr(e6r), pr.corr(IDX['沪深300'].pct_change().reindex(dates).fillna(0))))

print('\n=== 分年（%，逐日复利）===')
rows = []
for y in sorted(set(dates.year)):
    m = pr.index.year == y
    if m.sum() < 100: continue
    rows.append((y, ((1 + pr[m]).prod() - 1) * 100, ((1 + e6r[m]).prod() - 1) * 100,
                 ((1 + pr[m]).prod() - (1 + e6r[m]).prod()) * 100,
                 pr[m].std() * PPY ** .5 * 100))
by = pd.DataFrame(rows, columns=['year', 'A2', 'ew60', 'ex', 'vol'])
print(by.round(2).to_string(index=False))
by.to_csv(f'{ROOT}/reports/A2_daily_byyear.csv', index=False)

def to_m(s):
    return (1 + s).groupby(pd.PeriodIndex(s.index, freq='M')).prod() - 1
pm, em = to_m(pr), to_m(e6r)
pm.name, em.name = 'A2', 'EW60'
pd.to_pickle(pm, f'{ROOT}/data/big/a2_monthly.pkl')
pd.to_pickle(em, f'{ROOT}/data/big/ew60_monthly.pkl')
print('\n已存 data/big/a2_monthly.pkl（%d 期）' % len(pm))
