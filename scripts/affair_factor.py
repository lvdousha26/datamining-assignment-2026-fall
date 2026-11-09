# -*- coding: utf-8 -*-
"""「婚外情反弹」因子检验 —— 标准事件研究（Event Study）

[已归档] 本脚本是第一次手写的事件研究，已被 scripts/event_study.py 取代。
    保留原因：它是「安慰剂分位」这个方法的起点，且数字可与新框架对照。
    注意：本脚本安慰剂分位的实现有 bug（MSFT 报 52%，正确值 95.6%），
    以 event_study.py 的结果为准。


待检验命题（网络梗）：
    AMZN 自 2019 贝索斯婚外情曝光后市值 +2 万亿美元
    MSFT 自 2021 盖茨出轨曝光后市值 +2 万亿美元
    → 「CEO 出丑 = 利好」

关键口径：必须用**前复权**价。AMZN 2022 年做过 20:1 拆股，
未复权价会算出「至今 −84%」这种荒谬结果（第一版就踩了这个坑）。

检验：
  ① 绝对涨幅
  ② 超额 CAR（减纳指）—— 排除「整个市场都在涨」
  ③ 安慰剂检验：同一只股票历史上所有同长度窗口的 CAR 分布，看事件窗口在第几百分位
  ④ 同侪对照：同日起算的其他大盘股 250 日 CAR
"""
import os, json
import numpy as np, pandas as pd
import akshare as ak

CACHE = 'D:/quant/data/us'
os.makedirs(CACHE, exist_ok=True)
MAPF = f'{CACHE}/_prefix.json'
WIN = [1, 5, 20, 60, 250, 500]
BENCH = 'IXIC'
EVENTS = [('AMZN', '2019-01-09', '贝索斯婚外情曝光（National Enquirer）'),
          ('MSFT', '2021-05-16', '盖茨离婚/董事会调查曝光（WSJ）')]
PEERS = ['AAPL', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'WMT', 'XOM', 'V']

PMAP = json.load(open(MAPF)) if os.path.exists(MAPF) else {}


def resolve(tk):
    if tk in PMAP: return PMAP[tk]
    for p in ['105', '106', '107']:
        try:
            d = ak.stock_us_hist(symbol=f'{p}.{tk}', period='daily',
                                 start_date='20260101', end_date='20261107', adjust='qfq')
            if len(d):
                PMAP[tk] = f'{p}.{tk}'; json.dump(PMAP, open(MAPF, 'w'))
                return PMAP[tk]
        except Exception:
            pass
    raise RuntimeError('无法解析代码 ' + tk)


def load(tk):
    f = f'{CACHE}/{tk}_hfq.parquet'
    if os.path.exists(f):
        return pd.read_parquet(f)['close']
    if tk == BENCH:
        d = ak.index_us_stock_sina(symbol='.IXIC')
        d['date'] = pd.to_datetime(d['date'])
        s = d.set_index('date')['close'].sort_index().dropna()
    else:
        code = resolve(tk)
        # 必须用 hfq（后复权）：qfq 在东财美股接口上会把历史价格调成负数
        # （JNJ 最小价 −19.60、JPM −32.31），算出的收益完全不可用。
        d = ak.stock_us_hist(symbol=code, period='daily', start_date='20040101',
                             end_date='20261107', adjust='hfq')
        d['date'] = pd.to_datetime(d['日期'])
        s = d.set_index('date')['收盘'].astype(float).sort_index().dropna()
        s = s[s > 0]
        assert s.min() > 0, f'{tk} 出现非正价格，复权数据异常'
    s.to_frame('close').to_parquet(f)
    return s


print('拉取数据…', flush=True)
px = {s: load(s) for s in ['AMZN', 'MSFT', BENCH]}
bench = px[BENCH]
print('纳指 %s ~ %s；AMZN %d 行；MSFT %d 行' %
      (bench.index[0].date(), bench.index[-1].date(), len(px['AMZN']), len(px['MSFT'])), flush=True)

for tk, dt, desc in EVENTS:
    df = pd.concat([px[tk].rename('s'), bench.rename('b')], axis=1, sort=True).dropna()
    A, B = df.s.values.astype(float), df.b.values.astype(float)
    i0 = int(np.searchsorted(df.index.values, np.datetime64(dt)))
    print('\n' + '=' * 76)
    print('%s  事件日 %s（%s）' % (tk, df.index[i0].date(), desc))
    print('=' * 76)
    print('%-14s %9s %9s %9s %9s %8s' % ('窗口', '个股', '纳指', '超额CAR', 'CAR年化', '安慰剂分位'))
    for w in WIN:
        if i0 + w >= len(A): continue
        r = A[i0 + w] / A[i0] - 1
        rb = B[i0 + w] / B[i0] - 1
        car = (1 + r) / (1 + rb) - 1
        ann = (1 + car) ** (252 / w) - 1
        rs = A[w:] / A[:-w] - 1
        rr = B[w:] / B[:-w] - 1
        cars = (1 + rs) / (1 + rr) - 1
        pct = (cars < car).mean() * 100
        print('%-14s %+8.1f%% %+8.1f%% %+8.1f%% %+8.1f%% %7.0f%%' %
              ('+%d 日' % w, r * 100, rb * 100, car * 100, ann * 100, pct))
    r = A[-1] / A[i0] - 1; rb = B[-1] / B[i0] - 1
    print('%-14s %+8.1f%% %+8.1f%% %+8.1f%%' % ('至今 %d 日' % (len(A) - 1 - i0), r * 100, rb * 100,
                                                ((1 + r) / (1 + rb) - 1) * 100))

print('\n' + '=' * 76)
print('同侪对照：事件日起 250 个交易日，同期超额 CAR')
print('=' * 76)
for tk, dt, _ in EVENTS:
    rows = []
    for p in PEERS + [tk]:
        try:
            q = load(p)
        except Exception as e:
            print('  %s 跳过 (%s)' % (p, str(e)[:40])); continue
        d2 = pd.concat([q.rename('s'), bench.rename('b')], axis=1, sort=True).dropna()
        i = int(np.searchsorted(d2.index.values, np.datetime64(dt)))
        if i + 250 >= len(d2): continue
        r = d2.s.values[i + 250] / d2.s.values[i] - 1
        rb = d2.b.values[i + 250] / d2.b.values[i] - 1
        rows.append((p, r * 100, ((1 + r) / (1 + rb) - 1) * 100))
    d = pd.DataFrame(rows, columns=['ticker', '250日涨幅%', '超额CAR%']).sort_values('超额CAR%')
    print('\n事件 %s（%s）' % (tk, dt))
    print(d.round(1).to_string(index=False))
    med = d[d.ticker != tk]['超额CAR%'].median()
    ev = float(d.loc[d.ticker == tk, '超额CAR%'].iloc[0])
    print('  → 事件股超额 %+.1f%%，同侪中位数 %+.1f%%，差 %+.1fpp' % (ev, med, ev - med))
