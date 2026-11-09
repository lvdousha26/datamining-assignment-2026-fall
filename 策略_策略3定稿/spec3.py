# -*- coding: utf-8 -*-
"""策略③ 定稿：A股腿 = 红利低波50% + 上证丹尼斯50%，其余四腿不变
口径A: 2019-01起（真实512890，修补拆分+2.5%分红估）
口径B: 2007-02起（用中证红利全收益H00922代理红利低波，用于长周期检验）
"""
import io, contextlib, json, subprocess
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

with contextlib.redirect_stdout(io.StringIO()):
    import tune5
    from tune import yahoo, sina, DIV as DIVIDX

def k(sym):
    u = ('https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
         '?symbol=%s&scale=240&ma=no&datalen=2000' % sym)
    d = json.loads(subprocess.run(['curl', '-s', '--max-time', '25', '-H', 'User-Agent: Mozilla/5.0', u],
                                  capture_output=True).stdout.decode())
    return pd.Series({pd.Timestamp(x['day']): float(x['close']) for x in d}).sort_index()

def fix(px, lim=-0.15):
    r = px.pct_change(); bad = r[r < lim].index
    s = np.array(px.values, float, copy=True)
    for d in bad: s[px.index.get_loc(d):] *= 1 / (1 + r.loc[d])
    return pd.Series(s, index=px.index)

def pm(px):
    r = (1 + px.pct_change()).resample('ME').prod() - 1
    r.index = r.index.to_period('M'); return r

DL = fix(k('sh512890'))
legs = {}
def leg(x):
    s = pd.Series(x[tune5.in_win], index=tune5.ALL_ME[tune5.in_win]).copy()
    s.index = s.index.to_period('M'); return s
legs['mom'] = leg(tune5.M_mom[10]); legs['ma'] = leg(tune5.mat(tune5.ma_mret(125)))
legs['don'] = leg(tune5.M_don[(90, 40)]); legs['brk'] = leg(tune5.BRK_arr)
legs['gold'] = pm(yahoo('y_GC=F.json')); legs['divlv'] = pm(DL) + 0.025 / 12
legs['div'] = pm(DIVIDX)

W3 = {'mom': .26, 'ma': .26, 'brk': .11, 'gold': .11}   # A股腿 26% 由 divlv/don 对半分
def build(w, half):
    out = w['mom'] * legs['mom'] + w['ma'] * legs['ma'] + w['brk'] * legs['brk'] + w['gold'] * legs['gold']
    return out + .13 * legs[half[0]] + .13 * legs[half[1]]
base1 = legs['mom'] * .26 + legs['ma'] * .26 + legs['don'] * .26 + legs['brk'] * .11 + legs['gold'] * .11

def stats(r, label):
    r = r.dropna(); nav = (1 + r).cumprod(); y = len(r) / 12
    a = nav.iloc[-1] ** (1 / y) - 1; v = r.std() * 12 ** .5
    print('  %-30s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%%' % (
        label, a * 100, v * 100, (a - .02) / v, (nav / nav.cummax() - 1).min() * 100))
    return a, v

print('=== 口径A：2019-01 起，真实红利低波ETF ===')
wA = legs['divlv'].dropna().index.intersection(legs['don'].dropna().index)
s3A = build(W3, ('divlv', 'don')).reindex(wA)
s1A = base1.reindex(wA)
stats(s1A, '① 原5腿（A股腿=丹尼斯）')
stats(s3A, '③ 红利低波50%+丹尼斯50%')

print('\n=== 口径B：2007-02 起，中证红利全收益代理 ===')
wB = legs['div'].dropna().index.intersection(legs['don'].dropna().index)
s3B = build(W3, ('div', 'don')).reindex(wB)
s1B = base1.reindex(wB)
stats(s1B, '① 原5腿（A股腿=丹尼斯）')
stats(s3B, '③ 红利50%+丹尼斯50%（代理）')

print('\n=== 口径B 分年度（③ vs ① vs 上证）===')
SSE = pm(tune5.SSE)
y3 = ((1 + s3B).groupby(s3B.index.year).prod() - 1) * 100
y1 = ((1 + s1B).groupby(s1B.index.year).prod() - 1) * 100
ys = SSE.reindex(s3B.index).fillna(0)
ys = ((1 + ys).groupby(ys.index.year).prod() - 1) * 100
for y in y3.index:
    print('  %d: ③ %+6.1f%%  ① %+6.1f%%  上证 %+6.1f%%' % (y, y3[y], y1[y], ys.get(y, np.nan)))

# ---- 当前目标仓位 ----
dev = tune5.dev.reindex(tune5.ndx_me).ffill().iloc[-1]
w_mom = min(1, max(0, 0.5 + 10 * dev))
w_ma = 1.0 if tune5.NDX.iloc[-1] > tune5.NDX.rolling(125).mean().iloc[-1] else 0.0
w_don = tune5.donch_state_m(90, 40).iloc[-1]
print('\n=== 当前应持仓位（信号日 %s）===' % tune5.ndx_me[-1].date())
pos = {'纳指趋势线动量': .26 * w_mom, '纳指MA125择时': .26 * w_ma,
       '红利低波(常持)': .13, '上证丹尼斯': .13 * w_don, '伯克希尔': .11, '黄金': .11}
for k2, v in pos.items(): print('  %-14s %5.1f%%' % (k2, v * 100))
print('  %-14s %5.1f%%  ← 现金/货基' % ('合计', sum(pos.values()) * 100))
print('  %.1f%% 空仓' % (100 - sum(pos.values()) * 100))

# ---- 图：双口径 ----
fig = plt.figure(figsize=(12, 10.5))
gs = fig.add_gridspec(3, 1, height_ratios=[4.5, 3.6, 2.2], hspace=0.34)
ax = fig.add_subplot(gs[0])
for ser, lab, c, lw in [(s3A, '③ 红利低波50%+丹尼斯50%', '#c0392b', 2.2),
                        (s1A, '① 原5腿', '#2980b9', 1.3),
                        (SSE.reindex(wA).fillna(0), '上证', '#7f8c8d', 1.0)]:
    nav = (1 + ser).cumprod()
    ax.plot(nav.index.to_timestamp('M'), nav.values, lw=lw, color=c, label=lab)
ax.set_yscale('log'); ax.legend(fontsize=10, loc='upper left', frameon=False)
ax.grid(alpha=.25, which='both')
ax.set_title('口径A：2019-01 起，真实红利低波ETF — ③ 夏普1.23 vs ① 1.05', fontsize=12)

ax2 = fig.add_subplot(gs[1])
for ser, lab, c, lw in [(s3B, '③ 用中证红利代理', '#c0392b', 1.6),
                        (s1B, '① 原5腿', '#2980b9', 1.2)]:
    nav = (1 + ser).cumprod()
    ax2.plot(nav.index.to_timestamp('M'), nav.values, lw=lw, color=c, label=lab)
ax2.set_yscale('log'); ax2.legend(fontsize=10, loc='upper left', frameon=False)
ax2.grid(alpha=.25, which='both')
ax2.set_title('口径B：2007-02 起，中证红利代理 — ③ 夏普0.90 回撤-22.5%（代理不是低波资产，仅供参考）', fontsize=11)

ax3 = fig.add_subplot(gs[2])
for ser, c in [(s3A, '#c0392b'), (s1A, '#2980b9')]:
    nav = (1 + ser).cumprod(); dd = (nav / nav.cummax() - 1) * 100
    ax3.plot(dd.index.to_timestamp('M'), dd.values, lw=1.2, color=c)
ax3.grid(alpha=.25); ax3.set_ylabel('回撤 %')
ax3.set_title('口径A 回撤：③ -12.4% vs ① -13.2%', fontsize=11)
plt.savefig('策略3_净值对比.png', dpi=135, bbox_inches='tight')
print('图已保存')
