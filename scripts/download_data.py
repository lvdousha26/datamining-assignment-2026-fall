# -*- coding: utf-8 -*-
"""下载 A 股日线（新浪源，前复权），随机抽样避免只取大市值
代码来自交易所官网全量列表（沪主板 + 深市），不依赖东财接口
输出：D:/quant/data/raw/{code}.csv
"""
import os, time, random
import pandas as pd
import akshare as ak

OUT = r'D:/quant/data/raw'
N = 260
START, END = '20190101', '20261031'
random.seed(2026)
os.makedirs(OUT, exist_ok=True)

codes = []
try:
    sh = ak.stock_info_sh_name_code(symbol="主板A股")
    codes += ['sh' + str(c).zfill(6) for c in sh['证券代码']]
except Exception as e:
    print('SH list fail', e)
try:
    sz = ak.stock_info_sz_name_code(symbol="A股列表")
    codes += ['sz' + str(c).zfill(6) for c in sz['A股代码']]
except Exception as e:
    print('SZ list fail', e)
codes = sorted(set(codes))
print('全量代码 %d 只' % len(codes))

pick = random.sample(codes, min(N, len(codes)))
ok = fail = 0
t0 = time.time()
for i, code in enumerate(pick, 1):
    fp = os.path.join(OUT, '%s.csv' % code)
    if os.path.exists(fp) and os.path.getsize(fp) > 3000:
        ok += 1; continue
    for attempt in range(2):
        try:
            df = ak.stock_zh_a_daily(symbol=code, start_date=START, end_date=END, adjust='qfq')
            if df is None or len(df) < 250:
                break
            df = df.rename(columns={'date': '日期', 'open': '开盘', 'high': '最高',
                                    'low': '最低', 'close': '收盘', 'amount': '成交额',
                                    'volume': '成交量'})
            df.to_csv(fp, index=False, encoding='utf-8-sig')
            ok += 1
            break
        except Exception:
            if attempt: fail += 1
            time.sleep(1.0)
    if i % 40 == 0:
        print('  %d/%d 成功%d 失败%d 用时%.0fs' % (i, len(pick), ok, fail, time.time()-t0), flush=True)

print('完成: 成功 %d 失败 %d，总用时 %.0fs' % (ok, fail, time.time()-t0))
