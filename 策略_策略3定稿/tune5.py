# -*- coding: utf-8 -*-
"""全因子网格搜索 + 速度评测 v2
修复: 1) 丹尼斯跌破下轨应清仓(0)而非延续仓位  2) 网格评估改为对齐后的纯numpy运算
网格: 纳指动量k(6) x 纳指MA(5) x 丹尼斯in(5) x out(5) x BRK权重(3) = 2250
"""
import json, time, itertools
import numpy as np, pandas as pd

COST = 5e-4
START = '2007-02-01'

def sina(f):
    d = json.load(open(f))
    t = pd.DataFrame([(x['day'], float(x['close'])) for x in d], columns=['d', 'c'])
    return t.drop_duplicates('d').set_index(pd.to_datetime(t['d'])).sort_index()['c'].astype(float)

def yahoo(f):
    d = json.load(open(f, encoding='utf-8'))['chart']['result'][0]
    t = pd.DataFrame({'d': pd.to_datetime(d['timestamp'], unit='s').normalize(),
                      'c': d['indicators']['quote'][0]['close']}).dropna()
    return t.drop_duplicates('d').set_index('d').sort_index()['c'].astype(float)

SSE = pd.read_pickle('sh_index.pkl')['close']
NDX = yahoo('y_%5EIXIC.json')
BRK = yahoo('y_BRK-B.json')

t0 = time.perf_counter()

def month_ends(idx):
    return pd.DatetimeIndex(sorted(set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last().values)))

# ---------- 纳指：月末序列 ----------
ndx_me = month_ends(NDX.index)
ndx_me_close = NDX.reindex(ndx_me).ffill()
ndx_mret = ndx_me_close.pct_change()  # 月收益

# 趋势线偏离度：回归只拟合一次，k是纯缩放
def line_dev(px, gap=252):
    idx = px.index; out = {}
    for d in month_ends(idx):
        loc = idx.get_loc(d); e = loc - gap
        if e < 252: continue
        y = px.iloc[:e+1].values; x = np.arange(len(y), dtype=float)
        a, b = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, y, rcond=None)[0]
        L = a + b * (len(y) - 1 + gap)
        out[d] = (px.iloc[loc] - L) / L
    return pd.Series(out).sort_index()

dev = line_dev(NDX)

# ---------- 上证：丹尼斯状态（月频） ----------
sse_me = month_ends(SSE.index)
sse_close = SSE.reindex(sse_me).ffill()
sse_mret = sse_close.pct_change()

def donch_state_m(n_in, n_out):
    """日频状态 -> 月末取样。突破n_in高持有(1)，跌破n_out低清仓(0)，之间维持"""
    hi = SSE.rolling(n_in).max().shift(1)
    lo = SSE.rolling(n_out).min().shift(1)
    st = pd.Series(np.nan, index=SSE.index)
    st[SSE > hi] = 1.0
    st[SSE < lo] = 0.0          # 修复点：跌破下轨是清仓信号
    st = st.ffill().fillna(0)
    return st.reindex(sse_me).ffill()

# ---------- 统一月频索引（从2006-12起对齐，保证shift正确） ----------
ALL_ME = month_ends(SSE.index.union(NDX.index).union(BRK.index))
ALL_ME = ALL_ME[ALL_ME >= pd.Timestamp('2006-06-01')]

def mat(series_m):
    """月收益序列 -> 对齐矩阵 (月份, 变体)"""
    return series_m.reindex(ALL_ME).fillna(0).values

def mom_mret(k):
    w = (0.5 + k * dev).clip(0, 1)
    w = w.reindex(ndx_me).ffill()
    return (w.shift(1) * ndx_mret).dropna()

def ma_mret(n):
    above = (NDX > NDX.rolling(n, min_periods=n // 2).mean()).astype(float)
    return (above.reindex(ndx_me).ffill().shift(1) * ndx_mret).dropna()

def don_mret(n_in, n_out):
    st = donch_state_m(n_in, n_out)
    return (st.shift(1) * sse_mret).dropna()

KS = [4, 6, 8, 10, 12, 14]
MANS = [100, 150, 200, 250, 300]
DINS = [30, 40, 55, 70, 90]
DOUTS = [10, 15, 20, 30, 40]
BRKWS = [0.0, 0.125, 0.25]

M_mom = {k: mat(mom_mret(k)) for k in KS}
M_ma = {n: mat(ma_mret(n)) for n in MANS}
M_don = {(i, o): mat(don_mret(i, o)) for i, o in itertools.product(DINS, DOUTS)}
brk_m = ((1 + BRK.pct_change()).resample('ME').prod() - 1)
BRK_arr = brk_m.reindex(ALL_ME).fillna(0).values
MONTHS = np.array([(d.year - ALL_ME[0].year) * 12 + (d.month - ALL_ME[0].month) for d in ALL_ME])
in_win = MONTHS >= ((pd.Timestamp(START).year - ALL_ME[0].year) * 12 + (pd.Timestamp(START).month - ALL_ME[0].month))

t_build = time.perf_counter() - t0

# ---------- 纯numpy网格评估 ----------
t1 = time.perf_counter()
def eval_all():
    out = []
    don_keys = list(M_don.keys())
    don_stack = np.stack([M_don[key] for key in don_keys])          # (25, T)
    mom_stack = np.stack([M_mom[k] for k in KS])                    # (6, T)
    ma_stack = np.stack([M_ma[n] for n in MANS])                    # (5, T)
    w = np.array([1/3, 1/3, 1/3])
    for bi, bw in enumerate(BRKWS):
        core_w = np.array([(1 - bw) / 3, (1 - bw) / 3, (1 - bw) / 3])
        for ma_v in ma_stack:
            for mom_v in mom_stack:
                # (25, T) 一次算完 25 个丹尼斯变体
                port = don_stack * core_w[2] + ma_v * core_w[1] + mom_v * core_w[0] + BRK_arr * bw
                P = port[:, in_win]                     # (25, T_win)
                nav = np.cumprod(1 + P, axis=1)
                T = P.shape[1]; yrs = T / 12
                anns = nav[:, -1] ** (1 / yrs) - 1
                vols = P.std(axis=1) * 12 ** .5
                mdds = (nav / np.maximum.accumulate(nav, axis=1) - 1).min(axis=1)
                h = T // 2
                def sh_batch(seg):
                    n = np.cumprod(1 + seg, axis=1)
                    a = n[:, -1] ** (1 / (seg.shape[1] / 12)) - 1
                    return (a - .02) / (seg.std(axis=1) * 12 ** .5)
                s1s = sh_batch(P[:, :h]); s2s = sh_batch(P[:, h:])
                for j in range(len(don_keys)):
                    out.append(((anns[j] - .02) / vols[j], anns[j], mdds[j], s1s[j], s2s[j]))
    return out

results_raw = eval_all()
t_grid = time.perf_counter() - t1
n_combos = len(results_raw)

# 附上参数标签
labels = [(k, n, di, do, bw) for bw in BRKWS for n in MANS for k in KS for (di, do) in M_don]
results = [(r[0], r[1], r[2], r[3], r[4], *lab) for r, lab in zip(results_raw, labels)]
results.sort(key=lambda x: -x[0])

print('信号构建: 36 个变体, 耗时 %.2f s' % t_build)
print('网格评估: %d 组合, 耗时 %.3f s -> %.0f 组合/秒 (numpy向量化)' % (n_combos, t_grid, n_combos / t_grid))
print('若朴素地每组重算全部信号: 约 %.0f s (%.1f 分钟) -> 慢 %d 倍' % (t_build * n_combos, t_build * n_combos / 60, t_build * n_combos / t_grid))

print('\n=== Top 12（全窗口夏普） ===')
print('%-6s %-4s %-5s %-4s %-4s %-6s %8s %8s %6s %6s' % ('夏普', 'k', 'MA', 'in', 'out', 'BRK', '年化', '回撤', '前半', '后半'))
for shp, ann, mdd, s1, s2, k, n, di, do, bw in results[:12]:
    print('%.3f  %-4d %-5d %-4d %-4d %-6s %7.2f%% %7.1f%% %6.2f %6.2f' % (
        shp, k, n, di, do, ('%.1f%%' % (bw * 100) if bw else '-'), ann * 100, mdd * 100, s1, s2))

cur = [r for r in results if r[5] == 10 and r[6] == 200 and r[7] == 55 and r[8] == 20 and abs(r[9] - 0.125) < 1e-9]
if cur:
    print('\n当前组合(k10/MA200/55-20/BRK12.5%%): 夏普 %.3f, 网格排名 %d / %d' % (cur[0][0], results.index(cur[0]) + 1, n_combos))
nb = [r for r in results if r[5] == 10 and r[6] == 200 and r[7] == 55 and r[8] == 20 and r[9] == 0]
if nb:
    print('同参数无BRK          : 夏普 %.3f (对照前面框架的 0.738)' % nb[0][0])

from collections import Counter
top100 = results[:100]
print('\nTop100 参数分布:')
for idx, nm in [(5, 'k'), (6, 'MA'), (7, 'in'), (8, 'out'), (9, 'BRK')]:
    print('  %-4s' % nm, dict(Counter(r[idx] for r in top100).most_common()))
