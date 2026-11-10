# -*- coding: utf-8 -*-
"""涨停股池每日落库（封单量 / 首封时间 / 炸板次数 / 连板数）

为什么单独存
------------
A 股唯一免费可得、又确实带「订单簿信息」的字段就是涨停封单量——
它是挂在涨停价上的真实委托，且不像盘中十档那样随时可撤。但它有两个硬约束：

1. **东财只保留最近约 15 个交易日**（实测 2026-10-20 返回空，10-21 起有数据）。
   所以这个因子**无法回测**，只能从今天开始逐日攒，攒够一年（约 240 日 × 50 只）
   才有统计意义。这正是本脚本存在的理由：每天收盘后跑一次，把易得变成已得。
2. 它天然稀疏（一天 50~100 只 / 5000 只），只能做「涨停内部的排序」或
   「次日溢价的择时」，不能塞进 42 因子的日频截面（覆盖率 <2%，进了也是 NaN 噪音）。

用法
----
  python scripts/fetch_zt.py                # 补最近 40 个自然日（已存跳过）
  python scripts/fetch_zt.py --days 120     # 多试一些，空表不落盘
产出
----
  data/big/zt/YYYY-MM-DD.parquet   每日一份
  data/big/zt_all.parquet          合并表（每次重写）

字段口径：封板资金 = 涨停价上未成交的买单金额（元）；seal_ratio = 封板资金 / 流通市值。
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
OUT = f'{ROOT}/data/big/zt'
os.makedirs(OUT, exist_ok=True)

RENAME = {
    '代码': 'symbol', '名称': 'name', '涨跌幅': 'pct_chg', '最新价': 'close',
    '成交额': 'amount', '流通市值': 'float_mv', '总市值': 'total_mv', '换手率': 'turn',
    '封板资金': 'seal_amt', '首次封板时间': 'first_seal', '最后封板时间': 'last_seal',
    '炸板次数': 'break_times', '涨停统计': 'zt_stat', '连板数': 'boards', '所属行业': 'industry',
}


def minutes_of(hhmmss):
    """092500 → 距 09:30 的分钟数；空/异常返回 nan"""
    s = str(hhmmss).strip()
    if len(s) < 6 or not s.isdigit():
        return np.nan
    h, m = int(s[:2]), int(s[2:4])
    return (h * 60 + m) - (9 * 60 + 30)


def one(date):
    import akshare as ak
    for k in range(3):
        try:
            df = ak.stock_zt_pool_em(date=date)
            break
        except Exception as e:
            if k == 2:
                return date, None, repr(e)
            time.sleep(1.5 * (k + 1))
    if df is None or len(df) == 0:
        return date, None, 'empty'
    df = df.rename(columns=RENAME)
    df = df[[c for c in RENAME.values() if c in df.columns]].copy()
    df['symbol'] = df['symbol'].astype(str).str.zfill(6)
    df['date'] = pd.to_datetime(date)
    for c in ['seal_amt', 'float_mv', 'total_mv', 'amount']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['seal_ratio'] = df['seal_amt'] / df['float_mv'].replace(0, np.nan)
    df['first_min'] = df['first_seal'].map(minutes_of)
    df['last_min'] = df['last_seal'].map(minutes_of)
    cols = ['date', 'symbol', 'name', 'pct_chg', 'close', 'amount', 'float_mv', 'turn',
            'seal_amt', 'seal_ratio', 'first_min', 'last_min', 'break_times', 'boards', 'industry']
    return date, df[[c for c in cols if c in df.columns]], 'ok'


def safe_write(df, path):
    """先写临时文件再 rename：Windows 上直接覆盖偶尔会被杀软/索引占用（WinError 5）"""
    tmp = path + '.tmp'
    for k in range(4):
        try:
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.8 * (k + 1))
        finally:
            if os.path.exists(tmp) and k == 3:
                os.remove(tmp)
    return False


def main(days=40):
    end = pd.Timestamp.today().normalize()
    cal = pd.bdate_range(end=end, periods=days)
    print('尝试区间 %s ~ %s（%d 个工作日）' % (cal[0].date(), cal[-1].date(), len(cal)))
    got, miss, empty = 0, [], 0
    for d in cal:
        date = d.strftime('%Y%m%d')
        f = f'{OUT}/{d.date()}.parquet'
        if os.path.exists(f) and os.path.getsize(f) > 0:
            got += 1
            continue
        _, df, status = one(date)
        if df is None:
            if status == 'empty':
                empty += 1
            else:
                miss.append((date, status))
            continue
        if not safe_write(df, f):
            miss.append((date, 'write-denied'))
            continue
        got += 1
        print('  %s %3d 只 | 封单/流通市值 中位 %.2f%%' %
              (d.date(), len(df), df['seal_ratio'].median() * 100), flush=True)

    parts = [pd.read_parquet(f'{OUT}/{x}') for x in sorted(os.listdir(OUT))
             if x.endswith('.parquet') and os.path.getsize(f'{OUT}/{x}') > 0]
    if parts:
        allz = pd.concat(parts, ignore_index=True).sort_values(['date', 'symbol'])
        allz.to_parquet(f'{ROOT}/data/big/zt_all.parquet', index=False)
        print('\n合并 %d 日 / %d 行 → data/big/zt_all.parquet' % (allz['date'].nunique(), len(allz)))
        print('可用区间 %s ~ %s' % (allz['date'].min().date(), allz['date'].max().date()))
        print('每日涨停只数 中位 %.0f | 封单强度 中位 %.2f%% | 连板占比 %.1f%%' % (
            allz.groupby('date').size().median(), allz['seal_ratio'].median() * 100,
            (allz['boards'] > 1).mean() * 100))
    print('\n新增 %d 日 | 空表（无涨停/非交易日）%d | 失败 %d' % (got, empty, len(miss)))
    for m in miss[:5]:
        print('  失败', m)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=40)
    a = ap.parse_args()
    with RunLogger(__file__, tag='zt'):
        main(a.days)
