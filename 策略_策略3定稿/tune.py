# -*- coding: utf-8 -*-
"""自动调参循环：参数网格 -> 候选池 -> 前向贪心选择，直到夏普无明显提升"""
import json, itertools, numpy as np, pandas as pd

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
CSI = sina('sina_000300.json')
HSI = yahoo('y_%5EHSI.json')
NDX = yahoo('y_%5EIXIC.json')
DIV = pd.read_csv('h00922.csv'); DIV['d'] = pd.to_datetime(DIV['日期'])
DIV = DIV.sort_values('d').set_index('d')['收盘'].astype(float)
PE = pd.read_csv('pe_沪深30.csv'); PE['d'] = pd.to_datetime(PE['日期'])
PE = PE.sort_values('d').set_index('d')['滚动市盈率']
PE.index = PE.index.to_period('M').to_timestamp('M')

def month_ends(idx):
    return pd.DatetimeIndex(sorted(set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last().values)))

# ---------------- 策略发生器（可调参） ----------------
def line_w(px, k=5.0, gap=252, sign=-1):
    idx = px.index; w = {}
    for d in month_ends(idx):
        loc = idx.get_loc(d); e = loc - gap
        if e < 252: continue
        y = px.iloc[:e+1].values; x = np.arange(len(y), dtype=float)
        a, b = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, y, rcond=None)[0]
        L = a + b * (len(y) - 1 + gap)
        w[d] = float(np.clip(0.5 + sign * k * (px.iloc[loc] - L) / L, 0, 1))
    return pd.Series(w).sort_index()

def donchian_w(px, n_in=55, n_out=20):
    hi, lo = px.rolling(n_in).max().shift(1), px.rolling(n_out).min().shift(1)
    sig = np.where(px > hi, 1.0, np.where(px < lo, 0.0, np.nan))
    return pd.Series(sig, index=px.index).ffill().fillna(0)

def pe_graham_w(px, pe_m, low=0.3, high=0.7, win=120, minp=36):
    pct = pe_m.rolling(win, min_periods=minp).apply(lambda x: (x <= x.iloc[-1]).mean(), raw=False)
    sig_m = pd.Series(np.where(pct < low, 1.0, np.where(pct > high, 0.0, np.nan)), index=pe_m.index).ffill()
    m_keys = pd.DatetimeIndex([pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
                               for y, m in px.groupby([px.index.year, px.index.month]).last().index])
    return pd.Series(sig_m.reindex(m_keys).values, index=m_keys).reindex(px.index).ffill().fillna(0)

def macd_w(px, fast=12, slow=26, sig=9):
    dif = px.ewm(span=fast).mean() - px.ewm(span=slow).mean()
    dea = dif.ewm(span=sig).mean()
    return pd.Series(np.where(dif > dea, 1.0, 0.0), index=px.index)

def ma_cross_w(px, fast=20, slow=120):
    return pd.Series(np.where(px.rolling(fast).mean() > px.rolling(slow).mean(), 1.0, 0.0), index=px.index)

def rsi_w(px, n=14, buy=40, sell=60):
    d = px.diff(); up = d.clip(lower=0).ewm(alpha=1/n).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    s = np.where(rsi < buy, 1.0, np.where(rsi > sell, 0.0, np.nan))
    return pd.Series(s, index=px.index).ffill().fillna(0)

def vol_target_w(px, tv=0.15, n=20):
    v = px.pct_change().rolling(n).std() * 252 ** .5
    return (tv / v).clip(0, 1).fillna(0)

def high52_w(px, thr=0.95):
    hi = px.rolling(252, min_periods=126).max()
    return pd.Series(np.where(px >= thr * hi, 1.0, 0.0), index=px.index)

def strat_ret(px, w):
    idx = px.index; ret = px.pct_change().fillna(0)
    if len(w) < len(idx) * 0.2:
        eff = {}
        for d, v in w.items():
            loc = idx.get_loc(d)
            if loc + 1 < len(idx): eff[idx[loc + 1]] = v
        wd = pd.Series(eff).sort_index().reindex(idx).ffill()
    else:
        wd = w.reindex(idx).ffill()
    dw = wd.diff().abs().fillna(wd.abs())
    pr = (wd.shift(1) * ret - dw * COST).dropna()
    return pr[pr.index >= START]

# ---------------- 候选池（参数网格） ----------------
pool = {}
for k in [3, 4, 5, 6, 7]:
    pool['上证趋势线k%d' % k] = strat_ret(SSE, line_w(SSE, k))
for ki, ko in [(55, 20), (40, 20), (55, 10), (30, 15)]:
    pool['上证丹尼斯%d/%d' % (ki, ko)] = strat_ret(SSE, donchian_w(SSE, ki, ko))
for lo, hi in [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]:
    pool['沪深300格雷厄姆%.1f/%.1f' % (lo, hi)] = strat_ret(CSI, pe_graham_w(CSI, PE, lo, hi))
for k in [4, 5, 6]:
    pool['恒生趋势线k%d' % k] = strat_ret(HSI, line_w(HSI, k))
for f, s in [(12, 26), (8, 17), (20, 40)]:
    pool['上证MACD%d/%d' % (f, s)] = strat_ret(SSE, macd_w(SSE, f, s))
for k in [6, 8, 10]:
    pool['纳指趋势线动量k%d' % k] = strat_ret(NDX, line_w(NDX, k, sign=+1))
for thr in [0.90, 0.95]:
    pool['纳指52周高点%.2f' % thr] = strat_ret(NDX, high52_w(NDX, thr))
for f, s in [(20, 120), (50, 200), (10, 60)]:
    pool['上证双均线%d/%d' % (f, s)] = strat_ret(SSE, ma_cross_w(SSE, f, s))
for b, s in [(30, 70), (40, 60)]:
    pool['上证RSI%d/%d' % (b, s)] = strat_ret(SSE, rsi_w(SSE, 14, b, s))
for tv in [0.12, 0.15, 0.20]:
    pool['上证波动率目标%.2f' % tv] = strat_ret(SSE, vol_target_w(SSE, tv))
pool['红利全收益'] = strat_ret(DIV, pd.Series(1.0, index=DIV.index))

union = sorted(set().union(*[set(s.index) for s in pool.values()]))
union = pd.DatetimeIndex([d for d in union if d >= pd.Timestamp(START)])
df = pd.DataFrame({k: v.reindex(union).fillna(0) for k, v in pool.items()})
mret = (1 + df).resample('ME').prod() - 1

SSE_bench = strat_ret(SSE, pd.Series(1.0, index=SSE.index)).reindex(union).fillna(0)
bench_m = (1 + SSE_bench).resample('ME').prod() - 1

print('候选池 %d 个，窗口 %s ~ %s（%.1f 年）' % (len(df.columns), df.index[0].date(), df.index[-1].date(), len(df) / 252))

def port_stats(cols, method='ew'):
    sub = mret[list(cols)]
    if method == 'ew':
        w = pd.Series(1 / len(cols), index=list(cols))
    else:
        v = df[list(cols)].std() * 252 ** .5
        w = (1 / v) / (1 / v).sum()
    p = (sub * w).sum(axis=1)
    yrs = len(p) / 12
    nav = (1 + p).cumprod()
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    vol = p.std() * 12 ** .5
    shp = (ann - .02) / vol
    mdd = (nav / nav.cummax() - 1).min()
    e = (p - bench_m).dropna()
    te = e.mean() / (e.std() / len(e) ** .5) * 12 ** .5
    return shp, ann, vol, mdd, te, p

# ---------------- 前向贪心选择循环 ----------------
print('\n=== 前向贪心选择（每步尝试加入一个候选，等权+逆波动率取优） ===')
selected = []
best_shp = -9
history = []
for step in range(8):
    step_best = None
    for name in df.columns:
        if name in selected: continue
        for method in ['ew', 'iv']:
            shp, ann, vol, mdd, te, p = port_stats(selected + [name], method)
            if step_best is None or shp > step_best[0]:
                step_best = (shp, name, method, ann, vol, mdd, te)
    if step_best is None: break
    gain = step_best[0] - best_shp
    print('第%d步: +%-22s [%s]  夏普 %.3f (年化 %.1f%%, 回撤 %.1f%%, 超额t %.2f)  提升 %+.3f'
          % (step + 1, step_best[1], step_best[2], step_best[0], step_best[3] * 100,
             step_best[5] * 100, step_best[6], gain))
    if gain < 0.015:
        print('-> 提升不足 0.015，停止（不加入该策略）')
        break
    selected.append(step_best[1]); best_shp = step_best[0]
    history.append((step_best[1], step_best[0]))

print('\n最终组合:', selected)
shp, ann, vol, mdd, te, p = port_stats(selected, 'ew')
print('等权: 年化 %.2f%%  波动 %.2f%%  夏普 %.3f  回撤 %.2f%%  vs上证超额t %.2f' % (ann*100, vol*100, shp, mdd*100, te))

# ---------------- 分段稳健性 ----------------
half = len(p) // 2
for tag, seg in [('前半', p.iloc[:half]), ('后半', p.iloc[half:])]:
    yrs = len(seg) / 12; nav = (1 + seg).cumprod()
    a = nav.iloc[-1] ** (1 / yrs) - 1; v = seg.std() * 12 ** .5
    d = (nav / nav.cummax() - 1).min()
    print('%s (%s~%s): 年化 %.2f%%  夏普 %.3f  回撤 %.2f%%'
          % (tag, seg.index[0].date(), seg.index[-1].date(), a*100, (a-.02)/v, d*100))

# ---------------- 与旧基准对比 ----------------
print('\n=== vs 上一版最优（趋势线+丹尼斯+格雷厄姆+纳指动量, 夏普 0.58） ===')
old = ['上证趋势线k5', '上证丹尼斯55/20', '沪深300格雷厄姆0.3/0.7', '纳指趋势线动量k8']
avail = [c for c in old if c in df.columns]
print('旧组合列匹配:', avail)
shp0, ann0, vol0, mdd0, te0, p0 = port_stats(avail, 'ew')
print('旧组合复现: 年化 %.2f%%  夏普 %.3f  回撤 %.2f%%' % (ann0*100, shp0, mdd0*100))
print('新组合提升: 夏普 %+.3f  回撤 %+.1fpp' % (shp - shp0, (mdd - mdd0) * 100))

# 成分清单及权重建议（逆波动率）
v = df[selected].std() * 252 ** .5
w = (1 / v) / (1 / v).sum()
print('\n成分逆波动率权重参考:', {k: '%.0f%%' % (x * 100) for k, x in w.items()})
