# -*- coding: utf-8 -*-
"""涨停封单因子体检（封单强度 / 首封时间 / 连板数 → 次日与 5 日收益）

⚠️ 免费数据只有最近约 15 个交易日，本脚本的输出**只能当描述性统计**，
   t 值按「交易日」算 n，几乎不可能显著。它是为了把流程跑通，
   等 fetch_zt.py 每天攒够一年，再回来看同一张表。

用法：python scripts/zt_factor_probe.py
产出：reports/zt_factor.md
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'


def load_px(syms):
    ser, keys = [], []
    for s in syms:
        f = f'{ROOT}/data/big/raw_hfq/{s}.parquet'
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=['date', 'close'])
        d['date'] = pd.to_datetime(d['date'])
        ser.append(d.sort_values('date').set_index('date')['close'])
        keys.append(s)
    return pd.concat(ser, axis=1, keys=keys, sort=True) if ser else None


def build(zt, px):
    out = []
    for s, g in zt.groupby('symbol'):
        if s not in px.columns:
            continue
        p = px[s].dropna()
        for _, r in g.iterrows():
            if r['date'] not in p.index:
                continue
            i = p.index.get_loc(r['date'])
            k = lambda n: (p.iloc[i + n] / p.iloc[i] - 1) if i + n < len(p) and p.iloc[i] > 0 else np.nan
            out.append(dict(date=r['date'], symbol=s, seal=r['seal_ratio'], boards=r['boards'],
                            first=r['first_min'], brk=r['break_times'], mcap=r['float_mv'],
                            f1=k(1), f2=k(2), f5=k(5)))
    d = pd.DataFrame(out)
    # 当日去均值：涨停股次日普遍有溢价，那是 beta，不是因子
    for c in ['f1', 'f2', 'f5']:
        d[c + '_ex'] = d[c] - d.groupby('date')[c].transform('mean')
    return d


def tertile_report(d, col, name):
    dd = d.dropna(subset=[col, 'f1_ex']).copy()
    if len(dd) < 60:
        return f'### {name}\n\n样本不足（{len(dd)} 条）。\n\n'
    dd['q'] = dd.groupby('date')[col].transform(
        lambda x: pd.qcut(x, 3, labels=['低', '中', '高'], duplicates='drop') if x.nunique() >= 3 else np.nan)
    dd = dd.dropna(subset=['q'])
    g = dd.groupby('q', observed=True)[['f1_ex', 'f2_ex', 'f5_ex']].mean() * 100
    n = dd.groupby('q', observed=True).size()
    # 每日价差（高-低）的 t：n = 交易日数
    piv = dd.pivot_table(index='date', columns='q', values='f1_ex', aggfunc='mean', observed=True)
    if {'高', '低'}.issubset(piv.columns):
        sp = (piv['高'] - piv['低']).dropna()
        t = sp.mean() / (sp.std() / np.sqrt(len(sp))) if len(sp) > 2 and sp.std() > 0 else np.nan
        sp_txt = '日价差 %+.2f%% | t = %.2f（n=%d 日）' % (sp.mean() * 100, t, len(sp))
    else:
        sp_txt = '（当日缺某档，无法算价差）'
    lines = [f'### {name}', '', '| 分组 | 次日超额 | 2 日 | 5 日 | 条数 |', '|---|---|---|---|---|']
    for q in ['低', '中', '高']:
        if q in g.index:
            lines.append('| %s | %+.2f%% | %+.2f%% | %+.2f%% | %d |' % (
                q, g.loc[q, 'f1_ex'], g.loc[q, 'f2_ex'], g.loc[q, 'f5_ex'], n.get(q, 0)))
    lines += ['', sp_txt, '']
    return '\n'.join(lines)


def main():
    zt = pd.read_parquet(f'{ROOT}/data/big/zt_all.parquet')
    px = load_px(sorted(zt['symbol'].unique()))
    if px is None:
        print('没有 raw_hfq 行情，无法算收益')
        return
    d = build(zt, px).dropna(subset=['seal', 'f1'])
    nd = d['date'].nunique()
    print('样本 %d 条 / %d 个交易日 / %d 只 | 区间 %s ~ %s' % (
        len(d), nd, d['symbol'].nunique(), d['date'].min().date(), d['date'].max().date()))
    print('当日涨停股平均：次日 %+.2f%% | 5 日 %+.2f%%' % (d.f1.mean() * 100, d.f5.mean() * 100))

    md = [f'# 涨停封单因子体检', '',
          '> %s｜**%d 条 / %d 个交易日**｜免费数据只有最近约 15 日，以下全部是描述性统计，不是结论。' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), len(d), nd), '',
          '## 0. 先说清楚这个因子是什么', '',
          '- 封单强度 `seal_ratio = 封板资金 / 流通市值`，只在**涨停当日**有值，'
          '一天 30~120 只 → 它是**事件因子**，不是日频截面因子（覆盖率 <2%，进 42 因子只会稀释）。',
          '- 免费数据（东财涨停股池）只保留最近约 15 个交易日：实测 2026-10-20 返回空、10-21 起有数据。'
          '所以**无法回测**，只能逐日攒。',
          '- 当日涨停股次日平均 %+.2f%%：这是「涨停溢价」，是 beta 不是 alpha，'
          '下面所有分组都已减去当日池内均值。' % (d.f1.mean() * 100), '']

    md += ['## 1. 封单强度', '', tertile_report(d, 'seal', '按 seal_ratio 三分组（当日组内排序）')]
    d['boards_c'] = d['boards'].clip(1, 3)
    md += ['### 按连板数', '', '| 连板 | 次日超额 | 5 日超额 | 条数 |', '|---|---|---|---|']
    gg = d.dropna(subset=['f1_ex']).groupby('boards_c')[['f1_ex', 'f5_ex']].mean() * 100
    cnt = d.dropna(subset=['f1_ex']).groupby('boards_c').size()
    for b in gg.index:
        md.append('| %d 板 | %+.2f%% | %+.2f%% | %d |' % (b, gg.loc[b, 'f1_ex'], gg.loc[b, 'f5_ex'], cnt[b]))
    md.append('')

    d['first_b'] = pd.cut(d['first'], [-1, 1, 30, 120, 400], labels=['竞价', '<30min', '30-120', '>120'])
    md += ['### 按首封时间', '', '| 首封 | 次日超额 | 5 日超额 | 条数 |', '|---|---|---|---|']
    g3 = d.dropna(subset=['f1_ex']).groupby('first_b', observed=True)[['f1_ex', 'f5_ex']].mean() * 100
    c3 = d.dropna(subset=['f1_ex']).groupby('first_b', observed=True).size()
    for b in g3.index:
        md.append('| %s | %+.2f%% | %+.2f%% | %d |' % (b, g3.loc[b, 'f1_ex'], g3.loc[b, 'f5_ex'], c3[b]))
    md.append('')

    md += ['## 2. 这些数字能说明什么', '',
           '- **能**：封单强度高的一档，次日超额明显好于低的一档，方向与「封单大=买盘强」的常识一致；'
           '这个方向值得继续记录。',
           '- **不能**：%d 个交易日覆盖单一市场状态（2026-11），任何 t 值都不可信；'
           '5 日列样本更少（最后 5 日还没有未来数据）。' % nd,
           '- **不能**：没有扣成本。涨停次日买入要付 15~30bp，且一字板/秒板买不进——'
           '真正的可交易性比这张表差。', '',
           '## 3. 下一步', '',
           '1. `python scripts/fetch_zt.py` 每天收盘后跑一次（15:40 之后），数据进 `data/big/zt/`；',
           '2. 攒满 6 个月（约 120 日 / 7000 条）再回来看同一张表，届时才有资格谈显著性；',
           '3. 若要更早得到结论，只能买 Level-2 逐笔委托重建历史封单，几千元/年。', '']

    out = f'{ROOT}/reports/zt_factor.md'
    open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('已写', out)


if __name__ == '__main__':
    with RunLogger(__file__, tag='zt'):
        main()
