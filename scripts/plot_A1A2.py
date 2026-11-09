# -*- coding: utf-8 -*-
"""A1(70%) + A2风险预算(30%) 组合：净值曲线 + 回撤 + 分年"""
import io, contextlib, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT, BASE = 'D:/quant', r'C:/Users/25097/WorkBuddy/2026-10-30-23-31-35'

src = open(f'{ROOT}/scripts/combine_A1_A2.py', encoding='utf-8').read()
exec(src.split('# ---------- A1 ----------')[0])

bp = pd.read_pickle(f'{BASE}/idx_000012.pkl')
bm = (1 + bp.pct_change()).resample('ME').prod() - 1
bm.index = bm.index.to_period('M')
os.chdir(BASE)
with contextlib.redirect_stdout(io.StringIO()):
    sys.path.insert(0, BASE)
    import spec3
os.chdir(ROOT)
g = spec3.legs['gold']; g.index = pd.PeriodIndex(g.index, freq='M')

j = pd.concat([a2_abs.rename('A2'), bm.rename('B'), g.rename('G')], axis=1).dropna()
rv = j.A2.rolling(6, min_periods=4).std() * 12 ** .5
w = (0.08 / rv).clip(0.2, 1.0).shift(1).fillna(0.5)
a2v = w * j.A2 + (1 - w) * (0.7 * j.B + 0.3 * j.G)
a1 = spec3.s3B.dropna()

jj = pd.concat([a1.rename('A1'), a2v.rename('A2v')], axis=1).dropna()
combo = 0.7 * jj.A1 + 0.3 * jj.A2v

# 基准：沪深300
def sina_abs(f):
    import json
    d = json.load(open(os.path.join(BASE, f)))
    t = pd.DataFrame([(x['day'], float(x['close'])) for x in d], columns=['d', 'c'])
    return t.drop_duplicates('d').set_index(pd.to_datetime(t['d'])).sort_index()['c']


hs = sina_abs('sina_000300.json')
hm = (1 + hs.pct_change()).resample('ME').prod() - 1
hm.index = hm.index.to_period('M')
hm = hm.reindex(combo.index).fillna(0)


def m(x):
    a = (1 + x).prod() ** (12 / len(x)) - 1; v = x.std() * 12 ** .5
    nav = (1 + x).cumprod(); dd = (nav / nav.cummax() - 1).min()
    return a, v, (a - .02) / v, dd


series = {'A1 策略③': (jj.A1, '#7f8c8d', 1.2, '-'),
          'A2 质量+估值（风险预算）': (jj.A2v, '#e67e22', 1.3, '-'),
          'A1 70% + A2 30%': (combo, '#c0392b', 2.4, '-'),
          '沪深300': (hm, '#95a5a6', 1.0, '--')}

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[2.6, 1.1, 1.1], hspace=0.32, wspace=0.22)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])

x = combo.index.to_timestamp('M')
for nm, (s, c, lw, ls) in series.items():
    nav = (1 + s).cumprod()
    a, v, sh, dd = m(s)
    ax1.plot(x, nav.values, lw=lw, color=c, ls=ls, label='%s  年化%.2f%% 夏普%.2f 回撤%.1f%%' % (nm, a * 100, sh, dd * 100))
ax1.set_yscale('log'); ax1.legend(frameon=False, fontsize=9, loc='upper left')
ax1.grid(alpha=.22, which='both'); ax1.set_ylabel('净值（对数）')
ax1.set_title('A1 70%% + A2 30%%：%s ~ %s（%d 个月，费后）' %
              (combo.index[0], combo.index[-1], len(combo)), fontsize=11)

for nm, (s, c, lw, ls) in series.items():
    nav = (1 + s).cumprod()
    ax2.plot(x, (nav / nav.cummax() - 1).values * 100, lw=lw * .9, color=c, ls=ls)
ax2.grid(alpha=.22); ax2.set_ylabel('回撤 %'); ax2.axhline(0, color='#bdc3c7', lw=.6)

yrs = sorted(set(combo.index.year))
data = {nm: [((1 + s[s.index.year == y]).prod() - 1) * 100 if (s.index.year == y).any() else 0
             for y in yrs] for nm, (s, *_ ) in series.items()}
ws = 0.2
for k, (nm, v) in enumerate(data.items()):
    ax3.bar(np.arange(len(yrs)) + (k - 1.5) * ws, v, width=ws,
            color=series[nm][1], label=nm, alpha=.9)
ax3.set_xticks(range(len(yrs))); ax3.set_xticklabels(yrs, fontsize=9)
ax3.axhline(0, color='#7f8c8d', lw=.8); ax3.grid(alpha=.2, axis='y')
ax3.set_ylabel('年度收益 %'); ax3.legend(frameon=False, fontsize=8, ncol=4, loc='lower left')

plt.savefig(f'{ROOT}/reports/A1A2_combo.png', dpi=140, bbox_inches='tight')
print('图已存 reports/A1A2_combo.png')

# 打印关键表
print('\n%-26s %9s %9s %8s %9s' % ('', '年化', '波动', '夏普', '回撤'))
for nm, (s, *_) in series.items():
    a, v, sh, dd = m(s)
    print('%-26s %+8.2f%% %8.2f%% %8.2f %8.1f%%' % (nm, a * 100, v * 100, sh, dd * 100))

print('\n分年（%）：')
t = pd.DataFrame(data, index=yrs).round(1)
print(t.to_string())
t.to_csv(f'{ROOT}/reports/A1A2_combo_byyear.csv')
print('\n已存 reports/A1A2_combo_byyear.csv')
