# -*- coding: utf-8 -*-
"""全 A 股日线下载（akshare/东财，前复权）——规模扩到 5000+
存活偏差控制：当前名单 + 沪深两所退市名单（2015 年后退市的也拉进来）
分片 200 只落盘 + 断点续传
用法：python scripts/big_download.py [测试只数]
"""
import os, sys, time
import pandas as pd, akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings; warnings.filterwarnings('ignore')

ROOT = r'D:/quant'
OUT = f'{ROOT}/data/big/raw'
os.makedirs(OUT, exist_ok=True)
START = '20150101'
END = pd.Timestamp.today().strftime('%Y%m%d')   # 月度更新时自动取当日，不必手改
TEST = int(sys.argv[1]) if len(sys.argv) > 1 else 0

def universe():
    cur = ak.stock_zh_a_spot_em()[['代码', '名称']]
    cur = cur[cur['代码'].str.match(r'^(60|68|00|30)')][['代码']]
    cur.columns = ['code']
    out = [cur]
    for fn, cc, dd in [('stock_info_sh_delist', '公司代码', '暂停上市日期'),
                       ('stock_info_sz_delist', '证券代码', '终止上市日期')]:
        try:
            d = getattr(ak, fn)()[[cc, dd]]
            d.columns = ['code', 'delist']
            d = d[d['delist'].astype(str) >= '2015-01-01'][['code']]
            out.append(d)
        except Exception as e:
            print('  %s 失败 %s' % (fn, e), flush=True)
    u = pd.concat(out)['code'].astype(str).str.zfill(6).drop_duplicates().sort_values()
    u = u[u.str.match(r'^(60|68|00|30)')]
    return u.tolist()

def fetch(code):
    for attempt in range(3):
        try:
            d = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=START,
                                   end_date=END, adjust='qfq')
            if d is None or not len(d): return code, None
            d = d.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high',
                                  '最低': 'low', '成交量': 'volume', '成交额': 'amount',
                                  '振幅': 'amplitude', '涨跌幅': 'pctChg', '换手率': 'turn'})
            d['symbol'] = code
            d['date'] = pd.to_datetime(d['date'])
            return code, d[['date', 'symbol', 'open', 'high', 'low', 'close',
                            'volume', 'amount', 'amplitude', 'pctChg', 'turn']]
        except Exception:
            time.sleep(0.5 + attempt)
    return code, None

if __name__ == '__main__':
    codes = universe()
    print('universe %d 只（含退市）' % len(codes), flush=True)
    if TEST:
        codes = codes[:TEST]; print('测试模式 %d 只' % len(codes), flush=True)

    SH, t0, bad, done = 200, time.time(), [], 0
    shards = [codes[i:i + SH] for i in range(0, len(codes), SH)]
    for si, sh in enumerate(shards):
        fp = f'{OUT}/shard_{si:03d}.parquet'
        if os.path.exists(fp):
            done += len(sh); continue
        frames = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(fetch, c): c for c in sh}
            for f in as_completed(futs):
                c, d = f.result()
                if d is None: bad.append(c)
                else: frames.append(d)
        if frames: pd.concat(frames, ignore_index=True).to_parquet(fp, index=False)
        done += len(sh)
        print('  片 %d/%d 累计 %d 只 失败 %d 用时 %.0fs' % (
            si + 1, len(shards), done, len(bad), time.time() - t0), flush=True)
    print('完成 失败 %d：%s' % (len(bad), ','.join(bad[:30])), flush=True)
