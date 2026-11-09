# -*- coding: utf-8 -*-
"""从财务面板自动生成事件表（供 event_study.py 使用）

核心技巧：PIT 面板里财务数值是「公告日当天跳变」的，所以
    value(t) != value(t-1)  →  t 就是公告日
不需要单独维护公告日字段。

生成的事件类型
    earnings  财报惊喜：SUE = (本期同比 − 过去 4 期均值) / 过去 4 期标准差
              （标准 SUE 需要分析师预期，我们没有；用「标准化盈利变化」作代理，
                这是学术上可接受的替代，报告里必须说明）
    yjyg      业绩预告：预增/略增/扭亏/续盈 → 正面；预减/首亏/略减/续亏 → 负面
              标签是交易所强制披露的，比 NLP 猜情感可靠
    volume    放量异动：成交额 z-score > 阈值（不需要财务数据）

用法
    python scripts/make_events.py --type earnings --top 100
    python scripts/make_events.py --type earnings --top 100 --side neg   # 只取低于预期
    python scripts/make_events.py --type volume --top 200

输出
    events/earnings_surprise.csv（默认），列：symbol,date,label,note
"""
import argparse
import os

import numpy as np
import pandas as pd

# ---- 运行日志：统一写 logs/，不改变原有输出 ----
if __name__ == '__main__':
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from runlog import RunLogger as _RL
        _RL(__file__).__enter__()
    except Exception:
        pass

ROOT = r'D:/quant'


def earnings(top, side, thr, start='2016-01-01'):
    print('读财务面板…', flush=True)
    d = pd.read_parquet(f'{ROOT}/data/big/fin_panel.parquet',
                        columns=['date', 'symbol', 'profit_yoy', 'rev_yoy'])
    d = d[d['date'] >= pd.Timestamp(start)]
    d = d.sort_values(['symbol', 'date']).reset_index(drop=True)
    print(f'  {len(d):,} 行 / {d.symbol.nunique()} 只', flush=True)

    # 公告日 = profit_yoy 发生变化的那一天
    d['prev'] = d.groupby('symbol', sort=False)['profit_yoy'].shift(1)
    jump = d['profit_yoy'].notna() & (
        d['prev'].isna() | (d['profit_yoy'] - d['prev']).abs() > 1e-9)
    ev = d[jump].copy()
    print(f'  检出公告日 {len(ev):,} 个', flush=True)

    # SUE：本期值相对过去 4 期的标准化偏离
    ev = ev.sort_values(['symbol', 'date'])
    g = ev.groupby('symbol', sort=False)['profit_yoy']
    ev['mu'] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    ev['sd'] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=2).std())
    ev['sue'] = (ev['profit_yoy'] - ev['mu']) / (ev['sd'] + 1e-6)
    ev = ev[np.isfinite(ev['sue']) & (ev['sd'] > 0.05)]   # 分母过小会让 SUE 爆表
    print(f'  可算 SUE {len(ev):,} 个 | 中位 {ev.sue.median():.2f} | '
          f'p95 {ev.sue.quantile(.95):.2f} | p5 {ev.sue.quantile(.05):.2f}', flush=True)

    # 按年分层，避免极端值全部堆在早期（2016 年样本少、std 小）
    ev['yr'] = ev['date'].dt.year
    n_yr = max(int(ev[ev['sue'] >= thr]['yr'].nunique()), 1)
    per = max(top // n_yr, 3)

    def pick(df, asc):
        key = 'sue'
        return (df.groupby('yr', group_keys=False)
                  .apply(lambda g: g.nsmallest(per, key) if asc else g.nlargest(per, key),
                         include_groups=False)
                  .reset_index())

    if side in ('pos', 'both'):
        pos = pick(ev[ev['sue'] >= thr], False)
        pos['label'] = '财报超预期'
        pos['note'] = pos['sue'].map(lambda x: f'SUE={x:.2f}（代理口径）')
    else:
        pos = ev.iloc[0:0]
    if side in ('neg', 'both'):
        neg = pick(ev[ev['sue'] <= -thr], True)
        neg['label'] = '财报低预期'
        neg['note'] = neg['sue'].map(lambda x: f'SUE={x:.2f}（代理口径）')
    else:
        neg = ev.iloc[0:0]
    out = pd.concat([pos, neg])[['symbol', 'date', 'label', 'note']]
    return out.sort_values('date').reset_index(drop=True)


def yjyg(per_year):
    """业绩预告事件：标签直接来自预告类型"""
    POS = ['预增', '略增', '扭亏', '续盈']
    NEG = ['预减', '首亏', '略减', '续亏']
    d = pd.read_parquet(f'{ROOT}/data/big/yjyg/yjyg_all.parquet')
    d['date'] = pd.to_datetime(d['公告日期'])
    d['symbol'] = d['股票代码'].astype(str).str.zfill(6)
    d['label'] = d['预告类型'].map(lambda x: '业绩预喜' if x in POS else
                                   ('业绩预亏' if x in NEG else None))
    d = d[d['label'].notna() & (d['date'] >= pd.Timestamp('2016-01-01'))]
    # 同一股票同一公告日只留一条（不同预测指标会重复）
    d = d.drop_duplicates(['symbol', 'date', 'label'])
    d['yr'] = d['date'].dt.year
    d['abs_chg'] = pd.to_numeric(d['业绩变动幅度'], errors='coerce').abs().fillna(0)
    print(f'  预喜/预亏事件 {len(d):,} 个 | {d.symbol.nunique()} 只 | '
          f'{d.date.min().date()} ~ {d.date.max().date()}', flush=True)
    # 每年每类取变动幅度最大的 N 个
    out = (d.sort_values('abs_chg', ascending=False)
             .groupby(['yr', 'label']).head(per_year)
             .sort_values('date'))
    out['note'] = out['预告类型'] + '｜' + out['业绩变动'].astype(str).str[:36]
    return out[['symbol', 'date', 'label', 'note']].reset_index(drop=True)


def volume(top, thr):
    print('读日收益矩阵…', flush=True)
    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    R, dates = d['R'], pd.to_datetime(pd.Series(d['dates']).astype(str))
    syms = [str(s).zfill(6) for s in d['symbols']]
    vol = np.nanstd(R, axis=0)
    rows = []
    for j, s in enumerate(syms):
        r = R[:, j]
        if np.isfinite(r).sum() < 500:
            continue
        z = (np.abs(r) - np.nanmean(np.abs(r))) / (vol[j] + 1e-9)
        idx = np.where(np.isfinite(z) & (z > thr))[0]
        for i in idx[-50:]:                     # 每只股票最多 50 个
            rows.append((s, dates.iloc[i], '放量异动', f'|r|/std={z[i]:.1f}'))
    out = pd.DataFrame(rows, columns=['symbol', 'date', 'label', 'note'])
    if len(out) > top:
        out = out.sample(top, random_state=42).sort_values('date')
    return out.reset_index(drop=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--type', default='earnings', choices=['earnings', 'yjyg', 'volume'])
    ap.add_argument('--top', type=int, default=100)
    ap.add_argument('--side', default='both', choices=['pos', 'neg', 'both'])
    ap.add_argument('--thr', type=float, default=1.5, help='SUE 阈值 / 异动 z 阈值')
    ap.add_argument('--per-year', type=int, default=15, help='yjyg：每年每类取多少个')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    os.makedirs(f'{ROOT}/events', exist_ok=True)
    if a.type == 'earnings':
        ev = earnings(a.top, a.side, a.thr)
        out = a.out or f'{ROOT}/events/earnings_surprise.csv'
    elif a.type == 'yjyg':
        ev = yjyg(a.per_year)
        out = a.out or f'{ROOT}/events/yjyg.csv'
    else:
        ev = volume(a.top, a.thr)
        out = a.out or f'{ROOT}/events/volume_shock.csv'

    if len(ev) == 0:
        print('没有满足条件的事件，放宽 --thr 或提高 --top')
    else:
        ev['date'] = pd.to_datetime(ev['date']).dt.strftime('%Y-%m-%d')
        ev.to_csv(out, index=False)
        print(f'\n生成 {len(ev)} 个事件 → {out}')
        print(ev['label'].value_counts().to_string())
        print('\n按年分布：')
        print(pd.to_datetime(ev['date']).dt.year.value_counts().sort_index().to_string())
