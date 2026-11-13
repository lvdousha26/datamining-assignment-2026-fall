# -*- coding: utf-8 -*-
"""统一的绘图规范：中文、色盲友好、带数据来源脚注

作业要求（第 14 条）：中文坐标与单位、图题、数据来源、避免双纵轴、色盲可读。
这个模块把四项做成一个 `save()`，以后所有图都走它。

色板用 Okabe-Ito（8 色，色盲友好，红绿色盲也能区分），不用 matplotlib 默认的 jet/tab10。

用法：
    from figstyle import PALETTE, new_fig, save
    fig, ax = new_fig(1, 1)
    ax.plot(x, y, color=PALETTE[0])
    ax.set_xlabel('交易日（日）'); ax.set_ylabel('累计收益（%）')
    save(fig, 'reports/xxx.png', note='数据来源：data/big/daily_ret.npz（2016-2026，5,342 只）')
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Okabe-Ito 色板（色盲友好）
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
           '#E69F00', '#56B4E9', '#F0E442', '#000000']

plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def new_fig(nrow=1, ncol=1, w=4.6, h=3.6, **kw):
    """统一的画布：宽度按列数放大，高度按行数放大"""
    fig, ax = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), **kw)
    return fig, ax


def source_note(fig, note, y=0.005):
    """在图底部加数据来源脚注（小字号、左对齐）"""
    fig.text(0.01, y, note, fontsize=7.5, color='#555555', ha='left', va='bottom')


def save(fig, path, note=None, tight=True):
    if note:
        source_note(fig, note)
    if tight:
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path
