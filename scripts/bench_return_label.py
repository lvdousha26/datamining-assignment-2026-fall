# -*- coding: utf-8 -*-
"""换标签实验：用「公告次日超额收益」当标签，重跑同一批 3,000 条业绩预告

为什么值得做
------------
用「预告类型」当标签时，正文里写着增减，模型只需读数字——表格基线 93.2% 就能赢，
文本毫无增量。换成「次日超额收益」后，公告里的方向不再等于市场反应（预期差才是），
文本才第一次有机会提供**别的**信息。

口径
----
- 事件日：`公告日期`。t0 = 公告日当天或之前最近的交易日（盘后公告按当日算），
  t1 = 下一个交易日，反应收益 = R[t0→t1]（收盘到收盘）。
- 超额 = 个股次日收益 − 当日全市场等权收益（与 Y2 的基准口径一致）。
- 3 分类：超额 > +1% 利好；< −1% 利空；其余中性（同时报告二分类：涨/跌）。
- **按日期切分**（早 80% 训练 / 晚 20% 评估），不随机切——同一天的公告共享市场冲击，
  随机切会把评估集的信息漏进训练集。

用法：python scripts/bench_return_label.py
产出：runs/laya_yjyg/ret/{train.csv, eval.csv, stats.json} + reports/return_label.md
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
DIR = f'{ROOT}/runs/laya_yjyg'
OUT = f'{DIR}/ret'
THR = 0.01


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    R = d['R']
    dates = pd.to_datetime(pd.Series(d['dates']).astype(str))
    syms = [str(s).zfill(6) for s in d['symbols']]
    col = {s: j for j, s in enumerate(syms)}
    ew = np.nanmean(R, 1)

    # ---- 取回 3,000 条样本的公告日期（按 text 对齐，切分不变）----
    base = pd.concat([pd.read_csv(f'{DIR}/train.csv'), pd.read_csv(f'{DIR}/eval.csv')], ignore_index=True)
    fs = sorted(glob.glob(f'{ROOT}/data/big/yjyg/*.parquet'))
    y = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    y['text'] = (y['预测指标'].fillna('') + '｜' + y['业绩变动'].fillna('')).str.strip('｜')
    y['text'] = y['text'].str.replace(r'\s+', ' ', regex=True)
    y = y.drop_duplicates(subset=['text'], keep='first')
    m = base.merge(y[['text', '股票代码', '公告日期', '预告类型']], on='text', how='left')
    print('样本 %d | 匹配到公告 %d | 缺日期 %d' % (len(m), m['公告日期'].notna().sum(), m['公告日期'].isna().sum()))

    m['date'] = pd.to_datetime(m['公告日期'], errors='coerce')
    m['sym'] = m['股票代码'].astype(str).str.zfill(6)

    # ---- 次日超额 ----
    idx = dates.values
    t0 = np.searchsorted(idx, m['date'].values, side='right') - 1      # 公告日或之前最近的交易日
    t1 = t0 + 1
    ok = (t0 >= 0) & (t1 < len(dates)) & m['sym'].isin(col).values
    m = m[ok].copy()
    t1 = t1[ok]
    ci = np.array([col[s] for s in m['sym']])
    r = R[t1, ci]
    ex = r - ew[t1]
    m['r_next'] = r
    m['excess'] = ex
    m = m[np.isfinite(m['excess'])].copy()
    print('可用 %d 条 | 公告日 %s ~ %s' % (len(m), m['date'].min().date(), m['date'].max().date()))

    m['lab3'] = np.where(m.excess > THR, '利好', np.where(m.excess < -THR, '利空', '中性'))
    m['lab2'] = np.where(m.excess > 0, '涨', '跌')
    print('\n超额收益分布：均值 %.2f%% | 标准差 %.2f%% | 分位 %s' % (
        m.excess.mean() * 100, m.excess.std() * 100,
        np.round(np.percentile(m.excess, [5, 25, 50, 75, 95]) * 100, 2)))
    print('3 分类（±1%%）：%s' % m.lab3.value_counts().to_dict())
    print('2 分类（涨跌）：%s' % m.lab2.value_counts().to_dict())

    # 旧标签 vs 新标签 的一致性（关键证据）
    old = m['预告类型'].map(lambda t: '利好' if t in {'预增', '略增', '扭亏', '续盈', '减亏'}
                            else '利空' if t in {'预减', '略减', '首亏', '续亏', '增亏'} else '中性')
    m['lab_old'] = old
    agree = (old == m.lab3).mean()
    from scipy.stats import spearmanr
    rank_old = old.map({'利好': 1, '中性': 0, '利空': -1})
    rho = spearmanr(rank_old, m.excess).statistic
    print('\n旧标签（预告类型方向）与本标签（次日超额）一致率 **%.1f%%** | Spearman %.3f' % (agree * 100, rho))

    # ---- 按日期切分 ----
    m = m.sort_values('date').reset_index(drop=True)
    cut = int(len(m) * 0.8)
    tr, ev = m.iloc[:cut], m.iloc[cut:]
    print('训练 %d（%s ~ %s）| 评估 %d（%s ~ %s）' % (
        len(tr), tr.date.min().date(), tr.date.max().date(),
        len(ev), ev.date.min().date(), ev.date.max().date()))
    tr[['text', 'lab3', 'excess']].to_csv(f'{OUT}/train.csv', index=False)
    ev[['text', 'lab3', 'excess']].to_csv(f'{OUT}/eval.csv', index=False)

    json.dump({'n': len(m), 'train': len(tr), 'eval': len(ev),
               'agree_old_new': float(agree), 'spearman_old_excess': float(rho),
               'excess_std': float(m.excess.std()), 'thr': THR,
               'dist3': m.lab3.value_counts().to_dict(),
               'eval_dist3': ev.lab3.value_counts().to_dict(),
               'eval_majority': float(ev.lab3.value_counts(normalize=True).max())},
              open(f'{OUT}/stats.json', 'w'), ensure_ascii=False, indent=1)
    print('→', OUT)


if __name__ == '__main__':
    with RunLogger(__file__, tag='ret_label'):
        main()
