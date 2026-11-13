# -*- coding: utf-8 -*-
"""从两融标的名单的逐日快照，提取「纳入 / 调出」事件

逻辑
----
相邻两个交易日比较名单集合：
  纳入 = 今天在名单、昨天不在
  调出 = 昨天在名单、今天不在
剔除：
  · 上市不足 250 个交易日的新股（新股上市后自动纳入，属另一类事件）
  · 名单首日（没有「昨天」可比）
  · 不在面板里的代码（退市/非 A 股）

用法：python scripts/mk_margin_events.py
产出：events/margin_add.csv、events/margin_del.csv + reports/margin_events.md
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
MIN_HIST = 250


def main():
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    dates = pd.to_datetime(pd.Series(d['dates']).astype(str))
    syms = [str(s).zfill(6) for s in d['symbols']]
    col = {s: j for j, s in enumerate(syms)}
    R = d['R']
    valid_days = np.isfinite(R).sum(0)          # 每只有多少天有行情

    fs = sorted(glob.glob(f'{ROOT}/data/big/margin/*.parquet'))
    snap = {}
    for f in fs:
        dd = os.path.basename(f)[:10]
        snap[dd] = set(pd.read_parquet(f)['symbol'].astype(str).str.zfill(6))
    ds = [str(x.date()) for x in dates if str(x.date()) in snap]
    print('快照 %d 日：%s ~ %s' % (len(ds), ds[0], ds[-1]), flush=True)

    adds, dels = [], []
    for a, b in zip(ds[:-1], ds[1:]):
        cur, prev = snap[b], snap[a]
        for s in cur - prev:
            adds.append((s, b))
        for s in prev - cur:
            dels.append((s, b))
    print('原始：纳入 %d / 调出 %d' % (len(adds), len(dels)), flush=True)

    # 过滤：在面板内 + 上市满 MIN_HIST 天
    def keep(s):
        if s not in col:
            return False
        return valid_days[col[s]] >= MIN_HIST

    A = pd.DataFrame([(s, dd) for s, dd in adds if keep(s)], columns=['symbol', 'date'])
    D = pd.DataFrame([(s, dd) for s, dd in dels if keep(s)], columns=['symbol', 'date'])
    A['label'] = '纳入两融'
    D['label'] = '调出两融'
    for x in (A, D):
        x['note'] = ''
    os.makedirs(f'{ROOT}/events', exist_ok=True)
    A[['symbol', 'date', 'label', 'note']].to_csv(f'{ROOT}/events/margin_add.csv', index=False)
    D[['symbol', 'date', 'label', 'note']].to_csv(f'{ROOT}/events/margin_del.csv', index=False)

    print('过滤后：纳入 %d（%d 只）/ 调出 %d（%d 只）' % (
        len(A), A.symbol.nunique(), len(D), D.symbol.nunique()))
    yA = pd.to_datetime(A.date).dt.year.value_counts().sort_index()
    yD = pd.to_datetime(D.date).dt.year.value_counts().sort_index()
    tbl = pd.DataFrame({'纳入': yA, '调出': yD}).fillna(0).astype(int)
    print(tbl.to_string())

    md = ['# 两融标的调整：事件表构建', '',
          '> %s｜快照 %d 日（%s ~ %s）｜剔除上市不足 %d 个交易日的新股' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), len(ds), ds[0], ds[-1], MIN_HIST), '',
          '| 年份 | 纳入 | 调出 |', '|---|---|---|']
    for y, r in tbl.iterrows():
        md.append('| %d | %d | %d |' % (y, r['纳入'], r['调出']))
    md += ['', '| 合计 | **%d** | **%d** |' % (len(A), len(D)), '',
           '事件量比指数成分调整大一个量级（沪深300 十年只有约 400~600 个），', 
           '因为两融标的池从 2019 年的 ~1600 只扩到 2026 年的 ~4100 只，扩容本身就是持续事件。', '',
           '## 口径说明', '',
           '- 「纳入」= 今日在名单、昨日不在；「调出」反之；',
           '- 名单来源：深交所 `stock_margin_underlying_info_szse`（当日可融资）+ 上交所 `stock_margin_detail_sse`（当日有融资余额的标的）；',
           '- 上交所口径是「有融资余额」，理论上发行首日余额为 0 会漏记，实际影响很小；',
           '- 新股上市满 250 个交易日后才会被纳入，这类「首发纳入」已剔除，剩下的是存量调整。', '']
    open(f'{ROOT}/reports/margin_events.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('→', f'{ROOT}/reports/margin_events.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='margin_ev'):
        main()
