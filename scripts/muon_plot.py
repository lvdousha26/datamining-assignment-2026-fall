# -*- coding: utf-8 -*-
"""Muon A/B 结果可视化"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
ROOT = r'D:/quant'
df = pd.read_csv(f'{ROOT}/reports/muon_ab.csv')
cv = json.load(open(f'{ROOT}/reports/muon_curves.json'))
A = df[df.phase == 'A']; B = df[df.phase == 'B']

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Muon vs AdamW · Transformer 优化器对照实验', fontsize=16, fontweight='bold')

# --- A Phase A lr 扫描 ---
lbl = [('AdamW', '8e-4') if r.kind == 'adam' else ('Muon', r.lr) for _, r in A.iterrows()]
xs = ['AdamW\n8e-4' if r.kind == 'adam' else 'Muon\n%g' % r.lr for _, r in A.iterrows()]
cols = ['#95a5a6'] + ['#2980b9'] * 3
bars = ax[0, 0].bar(xs, A.ic22, color=cols, alpha=.9)
for b, (_, r) in zip(bars, A.iterrows()):
    ax[0, 0].text(b.get_x() + b.get_width() / 2, r.ic22 + .001, '%.4f' % r.ic22,
                  ha='center', fontsize=9)
ax[0, 0].set_ylim(0, max(A.ic22) * 1.2)
ax[0, 0].set_ylabel('2022 最佳 Rank IC')
ax[0, 0].set_title('A｜Phase A 学习率扫描（3ep / 1 seed，用 2022 选参）', fontsize=11)
ax[0, 0].grid(axis='y', alpha=.3)

# --- B Phase B 训练损失 ---
for kind, c, ls in [('AdamW', '#7f8c8d', '-'), ('Muon', '#2980b9', '-')]:
    ks = [k for k in cv if k.startswith(kind + '_s') and k.split('_s')[-1] in ('1', '2')]
    if not ks: continue
    n = min(len(cv[k]) for k in ks)
    L = np.mean([[cv[k][i]['loss'] for i in range(n)] for k in ks], 0)
    ax[0, 1].plot(range(1, n + 1), L, color=c, marker='o', lw=2, label='%s (2 seed 均值)' % kind)
ax[0, 1].set_xlabel('epoch'); ax[0, 1].set_ylabel('smooth L1 loss')
ax[0, 1].set_title('B｜Phase B 训练损失', fontsize=11)
ax[0, 1].legend(fontsize=9); ax[0, 1].grid(alpha=.3)

# --- C Phase B 逐 epoch 2022 IC ---
for kind, c in [('AdamW', '#7f8c8d'), ('Muon', '#2980b9')]:
    ks = [k for k in cv if k.startswith(kind + '_s') and k.split('_s')[-1] in ('1', '2')]
    if not ks: continue
    n = min(len(cv[k]) for k in ks)
    V = np.mean([[cv[k][i]['ic22'] for i in range(n)] for k in ks], 0)
    ax[1, 0].plot(range(1, n + 1), V, color=c, marker='s', lw=2, label=kind)
ax[1, 0].set_xlabel('epoch'); ax[1, 0].set_ylabel('2022 Rank IC')
ax[1, 0].set_title('C｜Phase B 验证集 IC 收敛过程', fontsize=11)
ax[1, 0].legend(fontsize=9); ax[1, 0].grid(alpha=.3)

# --- D 最终 OOS 对比 ---
metrics = ['ic23', 'ic24']
names = ['2023 验证', '2024+ 样本外']
x = np.arange(len(metrics)); w = .35
for j, (kind, c) in enumerate([('AdamW', '#7f8c8d'), ('Muon', '#2980b9')]):
    g = B[B.tag == kind]
    v = [g[m].mean() for m in metrics]
    e = [g[m].std() for m in metrics]
    bars = ax[1, 1].bar(x + (j - .5) * w, v, w, yerr=e, capsize=4,
                        color=c, alpha=.9, label=kind)
    for b, val in zip(bars, v):
        ax[1, 1].text(b.get_x() + b.get_width() / 2, val * .5, '%.4f' % val,
                      ha='center', va='center', fontsize=9,
                      color='white', fontweight='bold')
    for m in metrics:
        ax[1, 1].scatter(np.full(len(g), x[metrics.index(m)] + (j - .5) * w),
                         g[m], color=c, edgecolor='k', zorder=3, s=28)
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(names)
ax[1, 1].set_ylabel('Rank IC'); ax[1, 1].set_ylim(0, .14)
ax[1, 1].set_title('D｜最终泛化对比（柱=2 seed 均值，点=单 seed）', fontsize=11)
ax[1, 1].legend(fontsize=9); ax[1, 1].grid(axis='y', alpha=.3)

plt.tight_layout(rect=[0, 0, 1, .95])
plt.savefig(f'{ROOT}/reports/muon_ab.png', dpi=130)
print('已存 reports/muon_ab.png')
