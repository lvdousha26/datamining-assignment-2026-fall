# -*- coding: utf-8 -*-
"""质量+估值长线策略回测（复现"季度更新、换手<2倍、持有 200-400 只"的形态）

规则：
- 质量门槛：质量分（6 因子等权 z）前 40%
- 估值排序：池内按估值分（5 因子等权 z）取前 TOP 只
- 等权，每 60 个交易日（≈季度）调仓
- 持有期收益：12 个 5 日 Y2 窗口复利（Y2 已截面去均值 → 结果为相对全市场等权的超额）
- 成本：换手 × 2 × 15bp

对照：全市场等权 / 纯估值 top / 纯质量 top
"""
import numpy as np, pandas as pd

ROOT = 'D:/quant'
STEP, HOLD, TOP = 5, 60, 300
COST = 0.0015

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
Y2, M, dates = z['Y2'], z['M'], pd.to_datetime(z['dates'])
T, N = Y2.shape

fp = pd.read_parquet(f'{ROOT}/data/big/fin_panel.parquet')
fp['date'] = pd.to_datetime(fp['date'])
dm = {d: i for i, d in enumerate(dates)}
sm = {str(s).zfill(6): j for j, s in enumerate(z['symbols'])}
fp = fp[fp.date.isin(dm) & fp.symbol.isin(sm)].drop_duplicates(['date', 'symbol'])
ti = fp.date.map(dm).values.astype(np.int32)
si = fp.symbol.map(sm).values.astype(np.int32)

QUAL = ['roe_ttm', 'gpa', 'ocf_ni', 'accrual', 'debt_ratio', 'roe_stab']
QUA_DIR = np.array([1, 1, 1, -1, -1, -1])
VAL = ['bp', 'ep', 'sp', 'ocf_yield', 'fcf_yield']


def cz(v, t_idx):
    return pd.Series(v).groupby(t_idx).transform(lambda x: (x - x.mean()) / (x.std() + 1e-12)).values


# 质量分 / 估值分：逐因子截面 z → 等权平均（至少 3 / 2 个有效因子）
Q = np.zeros((T, N), np.float32); Qc = np.zeros((T, N), np.float32)
for k, c in enumerate(QUAL):
    zz = np.full((T, N), np.nan, np.float32)
    zz[ti, si] = cz(fp[c].values.astype(float), ti) * QUA_DIR[k]
    Q += np.nan_to_num(zz); Qc += np.isfinite(zz).astype(np.float32)
Q = np.where(Qc >= 3, Q / np.maximum(Qc, 1), np.nan)

Vv = np.zeros((T, N), np.float32); Vc = np.zeros((T, N), np.float32)
for c in VAL:
    zz = np.full((T, N), np.nan, np.float32)
    zz[ti, si] = cz(fp[c].values.astype(float), ti)
    Vv += np.nan_to_num(zz); Vc += np.isfinite(zz).astype(np.float32)
Vv = np.where(Vc >= 2, Vv / np.maximum(Vc, 1), np.nan)
print('质量/估值矩阵就绪', flush=True)

Q[~M] = np.nan; Vv[~M] = np.nan
eff = np.isfinite(Q) & np.isfinite(Vv) & M
print('有效格子 %d（%.1f%% of M）' % (eff.sum(), eff[M].mean() * 100), flush=True)


def pick(t, mode):
    """t 日收盘选股；返回股票索引"""
    m = eff[t]          # 只用当日可见信息（不可用 Y2 的 isfinite，那是未来函数）
    if m.sum() < 500: return None
    idx = np.where(m)[0]
    q, v = Q[t, idx], Vv[t, idx]
    if mode == 'all':
        return idx
    if mode == 'val':
        return idx[np.argsort(-v)[:TOP]]
    if mode == 'qua':
        return idx[np.argsort(-q)[:TOP]]
    thr = np.quantile(q, 0.60)                 # 质量前 40%
    pool = idx[q >= thr]
    if len(pool) < TOP: return None
    return pool[np.argsort(-Vv[t, pool])[:TOP]]


def backtest(mode):
    rows, prev = [], None
    for t0 in range(0, T - HOLD - STEP, HOLD):
        hold = pick(t0, mode)
        if hold is None or len(hold) < 50: continue
        # 持有期：12 个 5 日窗口复利
        rets = []
        for k in range(0, HOLD, STEP):
            tk = t0 + k
            y = Y2[tk, hold]
            rets.append(np.nanmean(y[np.isfinite(y)]))
        r = float(np.prod(1 + np.array(rets)) - 1)
        w = np.zeros(N); w[hold] = 1.0 / len(hold)
        to = 0.0 if prev is None else float(np.abs(w - prev).sum() / 2)
        rows.append((t0, r, to, len(hold)))
        prev = w
    d = pd.DataFrame(rows, columns=['t', 'r', 'to', 'n']).dropna(subset=['r'])
    d['net'] = d['r'] - d['to'] * 2 * COST
    d['yr'] = pd.DatetimeIndex(dates[d.t]).year
    return d


print('\n%-22s %9s %9s %7s %8s %8s' % ('策略（季度调仓·15bp）', '年化', '波动', '夏普', '最大回撤', '换手/年'))
res = {}
for nm, mode in [('全市场等权', 'all'), ('纯质量 top300', 'qua'),
                 ('纯估值 top300', 'val'), ('质量前40% → 估值top300', 'mix')]:
    d = backtest(mode)
    if len(d) < 5: print('%-22s 样本不足' % nm); continue
    ppy = 252 / HOLD
    ann = (1 + d.net).prod() ** (ppy / len(d)) - 1
    vol = d.net.std() * np.sqrt(ppy)
    cum = (1 + d.net).cumprod(); dd = (cum / cum.cummax() - 1).min()
    print('%-22s %+8.2f%% %8.2f%% %7.2f %7.1f%% %8.2f' %
          (nm, ann * 100, vol * 100, ann / vol if vol > 0 else 0, dd * 100, d.to.mean() * ppy * 2))
    res[nm] = d

print('\n=== 分年（净收益，%）===')
tab = pd.DataFrame({k: v.groupby('yr').apply(lambda g: ((1 + g.net).prod() - 1) * 100, include_groups=False)
                    for k, v in res.items()})
print(tab.round(1).to_string())

# 混合策略的年化序列，看趋势
if '质量前40% → 估值top300' in res:
    d = res['质量前40% → 估值top300']
    print('\n混合策略逐期明细（前 5 / 后 5）：')
    print(d.assign(date=dates[d.t].date)[['date', 'r', 'net', 'to', 'n']].head(5).round(4).to_string(index=False))
    print(d.assign(date=dates[d.t].date)[['date', 'r', 'net', 'to', 'n']].tail(5).round(4).to_string(index=False))
tab.round(2).to_csv(f'{ROOT}/reports/long_strategy_byyear.csv')
print('\n已存 reports/long_strategy_byyear.csv')
