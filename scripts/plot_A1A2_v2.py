# -*- coding: utf-8 -*-
"""A1 × A2 组合 v2 的图与诊断数字"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = 'D:/quant'
m = pd.read_pickle(f'{ROOT}/data/big/A1A2_v2_monthly.pkl')
j = m.dropna()
nav = (1 + j).cumprod()
dd = nav / nav.cummax() - 1

print('=== A2 超额（相对同频等权，已剔次新股）===')
ex = j.A2 - j.EW
print('  月均 %+.4f → 年化 %+.2f%% | 波动 %.2f%% | 夏普 %.2f'
      % (ex.mean() * 100, ex.mean() * 12 * 100, ex.std() * 12 ** .5 * 100,
         ex.mean() * 12 / (ex.std() * 12 ** .5)))
print('  正超额月份占比 %.0f%%' % ((ex > 0).mean() * 100))

# 对等权回归：alpha / beta
b = np.polyfit(j.EW, j.A2, 1)
print('  回归：beta %.2f | 月 alpha %+.4f（年化 %+.2f%%）| R² %.2f'
      % (b[0], b[1], b[1] * 12 * 100, np.corrcoef(j.EW, j.A2)[0, 1] ** 2))

print('\n=== 回撤区间 ===')
for c in ['A1', 'A2']:
    d = nav[c] / nav[c].cummax() - 1
    t = d.idxmin()
    pk = nav[c].loc[:t].idxmax()
    print('  %s 最大回撤 %.1f%%（%s → %s）' % (c, d.min() * 100, pk, t))

fig, ax = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw={'height_ratios': [3, 1.3, 1.5]})
cols = {'A1': '#4a4a4a', 'A2': '#e8871e', 'EW': '#9bb7d4'}
for c, lb in [('EW', '全市场等权（剔次新）'), ('A2', 'A2 质量+估值'), ('A1', 'A1 策略③')]:
    ax[0].plot(nav.index.to_timestamp(), nav[c], label=lb, color=cols[c],
               lw=2.2 if c == 'A1' else 1.6, zorder=3 if c == 'A1' else 2)
ax[0].set_yscale('log')
ax[0].set_title('A1 单独 vs A2 vs 全市场等权（绝对收益，已扣成本，2016-01~2026-10）', fontsize=13)
ax[0].legend(loc='upper left', frameon=False)
ax[0].grid(alpha=.25, ls='--')
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.2f}'))

for c in ['A1', 'A2']:
    ax[1].fill_between(dd.index.to_timestamp(), dd[c] * 100, 0,
                       color=cols[c], alpha=.35 if c == 'A2' else .55, label=c)
ax[1].set_title('回撤（%）', fontsize=11)
ax[1].legend(loc='lower left', frameon=False)
ax[1].grid(alpha=.25, ls='--')

by = pd.read_csv(f'{ROOT}/reports/A1A2_v2_byyear.csv')
x = np.arange(len(by)); w = 0.27
ax[2].bar(x - w, by.A1, w, label='A1', color=cols['A1'])
ax[2].bar(x, by.A2, w, label='A2', color=cols['A2'])
ax[2].bar(x + w, by['70/30'], w, label='70/30', color='#c0392b')
ax[2].set_xticks(x); ax[2].set_xticklabels(by.year, fontsize=9)
ax[2].axhline(0, color='k', lw=.8)
ax[2].set_title('分年净收益（%）', fontsize=11)
ax[2].legend(frameon=False, ncol=3)
ax[2].grid(alpha=.25, ls='--', axis='y')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.tight_layout()
plt.savefig(f'{ROOT}/reports/A1A2_v2.png', dpi=140)
print('\n已存 reports/A1A2_v2.png')
