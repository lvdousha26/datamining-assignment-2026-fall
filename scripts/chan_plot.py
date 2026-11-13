# -*- coding: utf-8 -*-
"""画一张缠论结构示意图：K 线合并 → 分型 → 笔 → 中枢

用法：python scripts/chan_plot.py [symbol] [窗口长度]
产出：reports/chan_structure.png
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import chan_factors as cf  # noqa: E402
from figstyle import PALETTE, save  # noqa: E402

ROOT = r'D:/quant'


def main(sym='600519', win=300):
    d = pd.read_parquet(f'{ROOT}/data/big/raw_hfq/{sym}.parquet')
    h = d['high'].values.astype(float)
    l = d['low'].values.astype(float)
    c = d['close'].values.astype(float)
    date = pd.to_datetime(d['date']).values

    mh, ml, mp = cf.merge_bars(h, l)
    M = len(mh)
    fx = cf.fractals(mh, ml)
    sts = cf.strokes(fx, mh, ml)
    zs = cf.pivots(sts)

    last_orig = np.zeros(M, np.int32)
    for i in range(len(c)):
        last_orig[mp[i]] = i

    i0 = max(len(c) - win, 0)
    x = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(x, c, color='#9aa0a6', lw=0.9, label='收盘价（后复权）')

    # 笔
    n_bi = 0
    for s in sts:
        a, b = last_orig[s[0]], last_orig[s[1]]
        if b < i0:
            continue
        ya = ml[s[0]] if s[2] == 1 else mh[s[0]]
        yb = mh[s[1]] if s[2] == 1 else ml[s[1]]
        ax.plot([a, b], [ya, yb], color=PALETTE[1], lw=1.6, alpha=.95)
        ax.scatter([b], [yb], s=14, color=PALETTE[1], zorder=5)
        n_bi += 1

    # 中枢
    n_zs = 0
    for z in zs:
        a, b = last_orig[sts[z[0]][0]], last_orig[sts[z[1]][1]]
        if b < i0:
            continue
        ax.add_patch(Rectangle((a, z[3]), b - a, z[2] - z[3], facecolor=PALETTE[2],
                               alpha=.18, edgecolor=PALETTE[2], lw=0.8))
        ax.plot([a, b], [z[2], z[2]], color=PALETTE[2], lw=0.7, ls='--')
        ax.plot([a, b], [z[3], z[3]], color=PALETTE[2], lw=0.7, ls='--')
        n_zs += 1

    ax.set_xlim(i0, len(c))
    ax.set_ylim(np.nanmin(l[i0:]) * 0.96, np.nanmax(h[i0:]) * 1.04)
    tk = np.linspace(i0, len(c) - 1, 7).astype(int)
    ax.set_xticks(tk)
    ax.set_xticklabels([pd.Timestamp(date[i]).strftime('%Y-%m') for i in tk], fontsize=8)
    ax.set_ylabel('价格（元，后复权）')
    ax.set_xlabel('日期（年-月）')
    ax.set_title('%s 缠论结构：笔与中枢（窗口 %d 日，%d 笔 / %d 中枢）' % (sym, win, n_bi, n_zs))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color='#9aa0a6', lw=0.9, label='收盘价'),
                       Line2D([], [], color=PALETTE[1], lw=1.6, label='笔'),
                       Line2D([], [], color=PALETTE[2], lw=3, alpha=.4, label='中枢（ZG~ZD）')],
              frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, f'{ROOT}/reports/chan_structure.png',
         note='数据来源：data/big/raw_hfq/%s.parquet；笔 = 顶底分型间隔≥4 根合并 K 线；'
              '中枢 = 连续三笔重叠区间（ZG=min 高点，ZD=max 低点）' % sym)
    print('→', f'{ROOT}/reports/chan_structure.png',
          '| 全序列 %d 笔 / %d 中枢，窗口内 %d / %d' % (len(sts), len(zs), n_bi, n_zs))


if __name__ == '__main__':
    s = sys.argv[1] if len(sys.argv) > 1 else '600519'
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    main(s, w)
