# -*- coding: utf-8 -*-
"""全市场财报数据下载（akshare 东财按报告期接口）

四张表：yjbb 业绩报表 / zcfz 资产负债表 / lrb 利润表 / xjll 现金流量表
范围：2013Q1 ~ 最新报告期（55 期）
特性：断点续传（每期每表落一个 parquet）、失败重试 3 次指数退避、限速
输出：data/big/fin/{table}_{YYYYMMDD}.parquet
"""
import os, time, sys, traceback
import pandas as pd
import akshare as ak

ROOT = 'D:/quant'
OUT = f'{ROOT}/data/big/fin'
os.makedirs(OUT, exist_ok=True)

TABLES = {'yjbb': ak.stock_yjbb_em, 'zcfz': ak.stock_zcfz_em,
          'lrb': ak.stock_lrb_em, 'xjll': ak.stock_xjll_em}

periods = []
for y in range(2013, 2027):
    for md in ['0331', '0630', '0930', '1231']:
        p = f'{y}{md}'
        if y == 2013 and md != '1231':
            periods.append(p)          # 保留 2013 全部，构造 12 季序列需要
        elif y > 2013:
            periods.append(p)
periods = [p for p in periods if p <= '20260930']
print('报告期 %d 个: %s ~ %s' % (len(periods), periods[0], periods[-1]), flush=True)

fail = []
t0 = time.time()
for i, p in enumerate(periods):
    for nm, fn in TABLES.items():
        f = f'{OUT}/{nm}_{p}.parquet'
        if os.path.exists(f) and os.path.getsize(f) > 2000:
            continue
        for attempt in range(3):
            try:
                df = fn(date=p)
                if df is None or len(df) == 0:
                    raise RuntimeError('empty')
                df['report_period'] = p
                df.to_parquet(f, index=False)
                print('[%d/%d] %s %-5s %5d 行  (%.0fs)' %
                      (i + 1, len(periods), p, nm, len(df), time.time() - t0), flush=True)
                break
            except Exception as e:
                if attempt == 2:
                    fail.append((p, nm, str(e)[:80]))
                    print('[%d/%d] %s %-5s FAIL %s' % (i + 1, len(periods), p, nm, str(e)[:60]), flush=True)
                else:
                    time.sleep(2 ** attempt * 2)
        time.sleep(0.4)

print('\n完成 %.1f 分钟，失败 %d 项' % ((time.time() - t0) / 60, len(fail)), flush=True)
if fail:
    pd.DataFrame(fail, columns=['period', 'table', 'err']).to_csv(f'{OUT}/_fail.csv', index=False)
    for f in fail[:10]: print('  ', f)
