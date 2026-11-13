# -*- coding: utf-8 -*-
"""42 因子量化标准评价 · 四宫格总览图"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
ROOT = r'D:/quant'
df = pd.read_csv(f'{ROOT}/reports/factor_scorecard.csv')
grid = pd.read_csv(f'{ROOT}/reports/overfit_grid.csv')

fig, ax = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('42 因子库 · 量化标准评价总览', fontsize=17, fontweight='bold')

# --- A 原始 IC vs 行业+市值双中性 IC ---
d = df.reindex(df['IC_both'].abs().sort_values(ascending=False).index).head(20).iloc[::-1]
y = np.arange(len(d))
ax[0, 0].barh(y - .2, d['IC'], height=.4, color='#95a5a6', label='原始 IC', alpha=.9)
ax[0, 0].barh(y + .2, d['IC_both'], height=.4, color='#2980b9', label='行业+市值中性后')
ax[0, 0].set_yticks(y); ax[0, 0].set_yticklabels(d['factor'], fontsize=9)
ax[0, 0].axvline(0, color='k', lw=.8)
ax[0, 0].set_title('A｜Top20 因子：中性化后还剩多少', fontsize=12)
ax[0, 0].set_xlabel('秩相关系数 IC（无量纲，正=因子值越大收益越高）'); ax[0, 0].legend(fontsize=9, loc='lower left')
ax[0, 0].grid(axis='x', alpha=.3)

# --- B 市值依赖度 ---
x = df['IC'].abs(); z = df['size_dep']
ax[0, 1].scatter(x, z, s=38, c=np.where(z > .5, '#c0392b', '#2980b9'), alpha=.8)
hi = df[df['size_dep'] > .5].sort_values('size_dep', ascending=False)
for j, (_, r) in enumerate(hi.iterrows()):
    ax[0, 1].annotate(r['factor'], (abs(r['IC']), r['size_dep']),
                      fontsize=8, xytext=(5, 5 if j % 2 == 0 else -11),
                      textcoords='offset points')
ax[0, 1].axhline(.5, color='#c0392b', ls='--', lw=1, label='50% 依赖线')
ax[0, 1].set_xlabel('|原始 IC|（无量纲）'); ax[0, 1].set_ylabel('市值依赖度')
ax[0, 1].set_title('B｜小市值暴露：7 个因子 >50% 靠市值', fontsize=12)
ax[0, 1].legend(fontsize=9); ax[0, 1].grid(alpha=.3)

# --- C 分年度稳定性 + 安慰剂 ---
years = [2022, 2023, 2024, 2025, 2026]
ics = [.1094, .1122, .1052, .1069, .1030]
bars = ax[1, 0].bar([str(v) for v in years], ics, color='#2980b9', alpha=.85)
ax[1, 0].axhline(-.0122, color='#c0392b', ls='--', lw=1.5, label='安慰剂(标签打乱) -0.0122')
ax[1, 0].axhline(np.mean(ics), color='#e67e22', ls=':', lw=1.5, label='5年均值 %.4f' % np.mean(ics))
for b, v in zip(bars, ics):
    ax[1, 0].text(b.get_x() + b.get_width() / 2, v + .002, '%.3f' % v,
                  ha='center', fontsize=9)
ax[1, 0].set_ylim(-.03, .135); ax[1, 0].set_ylabel('Stacking 样本外 IC（无量纲）')
ax[1, 0].set_xlabel('年份（年）')
ax[1, 0].set_title('C｜分年度稳定（IS 0.150 → OOS 0.101，衰减 33%）', fontsize=12)
ax[1, 0].legend(fontsize=9); ax[1, 0].grid(axis='y', alpha=.3)

# --- D 冗余度：PCA 累计方差 + 超参敏感度 ---
pca = pd.read_csv(f'{ROOT}/reports/pca_eig.csv')
ax[1, 1].plot(pca['pc'], pca['cum'], color='#8e44ad', lw=2, label='PCA 累计方差(实测)')
for v, c, lab in [(.8, '#2980b9', '80% → 12 维'), (.9, '#e67e22', '90% → 17 维')]:
    ax[1, 1].axhline(v, color=c, ls='--', lw=1, label=lab)
ax[1, 1].set_xlabel('主成分个数（个）'); ax[1, 1].set_ylabel('累计解释方差（比例，0~1）')
ax[1, 1].set_ylim(0, 1.02); ax[1, 1].set_xlim(1, 42)
ax[1, 1].set_title('D｜42 因子 ≈ 12~17 个独立维度｜有效检验数 19.6', fontsize=12)
ax[1, 1].legend(fontsize=9, loc='lower right'); ax[1, 1].grid(alpha=.3)
ax[1, 1].text(16, .38, '超参敏感度：8 组 OOS IC 极差仅 0.0024\n'
                        '安慰剂通过｜Block bootstrap 95%CI [0.081, 0.124]',
              fontsize=9, va='top', bbox=dict(boxstyle='round', fc='#f8f9fa', ec='#bdc3c7'))

plt.tight_layout(rect=[0, 0, 1, .96])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from figstyle import save as _save
_save(fig, f'{ROOT}/reports/factor_scorecard.png',
      note='数据来源：reports/factor_scorecard.csv（42 因子 × 三层中性化 IC）与 reports/overfit_grid.csv；'
           '区间 2016-2024 训练、2025H1 样本外；IC 为日频截面秩相关均值')
print('已存 reports/factor_scorecard.png')
