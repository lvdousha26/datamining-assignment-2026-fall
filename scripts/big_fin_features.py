# -*- coding: utf-8 -*-
"""PIT 基本面因子构建：质量四维 + 估值

流程：
1. 读 data/big/fin/ 四表 → 统一列名
2. 累计值转单季 → 滚动 4 季得 TTM
3. 按【公告日】铺到交易日（merge_asof，严格无未来函数）
4. 总市值 = (净利润累计 ÷ 每股收益累计) × 成交均价 —— 用 EPS 反推总股本，
   避免流通市值带来的"流通比例"偏差（次新股/国企大非会系统性失真）
5. 输出 data/big/fin_panel.parquet（date, symbol, 21 个因子）

质量  roe_ttm roa_ttm gpa gross_margin ocf_ni accrual debt_ratio asset_turn
      roe_stab roe_trend gm_trend rev_yoy profit_yoy recv_gap inv_gap ocf_yoy
估值  ep bp sp ocf_yield fcf_yield
"""
import glob
import numpy as np
import pandas as pd

ROOT = 'D:/quant'
FIN = f'{ROOT}/data/big/fin'
OUT = f'{ROOT}/data/big/fin_panel.parquet'


def load(t):
    fs = sorted(glob.glob(f'{FIN}/{t}_*.parquet'))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d.rename(columns={'股票代码': 'code', 'report_period': 'rp'})
    d['code'] = d['code'].astype(str).str.zfill(6)
    d['rp'] = pd.to_datetime(d['rp'], format='%Y%m%d')
    return d.drop_duplicates(['code', 'rp'], keep='last')


def legal_deadline(rp):
    """法定披露截止日：Q1→4/30, Q2→8/31, Q3→10/31, Q4→次年4/30"""
    y, m = rp.dt.year, rp.dt.month
    out = pd.Series(pd.NaT, index=rp.index)
    for mon, (mm, dd, add) in {3: (4, 30, 0), 6: (8, 31, 0), 9: (10, 31, 0), 12: (4, 30, 1)}.items():
        sel = m == mon
        if sel.any():
            out[sel] = pd.to_datetime(dict(year=y[sel] + add, month=mm, day=dd))
    return out


yj, zc, lr, xj = load('yjbb'), load('zcfz'), load('lrb'), load('xjll')
print('原始行数 yjbb %d / zcfz %d / lrb %d / xjll %d' % (len(yj), len(zc), len(lr), len(xj)), flush=True)

yj['ann'] = pd.to_datetime(yj['最新公告日期'], errors='coerce').fillna(legal_deadline(yj['rp']))
for d in (zc, lr, xj):
    d['ann'] = pd.to_datetime(d['公告日期'], errors='coerce').fillna(legal_deadline(d['rp']))

yj = yj.rename(columns={'营业总收入-营业总收入': 'rev', '净利润-净利润': 'ni', '每股收益': 'eps',
                        '每股净资产': 'bps', '净资产收益率': 'roe_rpt', '销售毛利率': 'gm',
                        '营业总收入-同比增长': 'rev_yoy_rpt', '净利润-同比增长': 'profit_yoy_rpt',
                        '所处行业': 'sw_ind'})
yj = yj[['code', 'rp', 'ann', 'rev', 'ni', 'eps', 'bps', 'roe_rpt', 'gm',
         'rev_yoy_rpt', 'profit_yoy_rpt', 'sw_ind']]

zc = zc.rename(columns={'资产-总资产': 'ta', '负债-总负债': 'tl', '股东权益合计': 'eq',
                        '资产-应收账款': 'recv', '资产-存货': 'inv', '资产负债率': 'debt_ratio'})
zc = zc[['code', 'rp', 'ann', 'ta', 'tl', 'eq', 'recv', 'inv', 'debt_ratio']]

lr = lr.rename(columns={'净利润': 'ni2', '营业总收入': 'rev2', '营业总支出-营业支出': 'cogs'})
lr = lr[['code', 'rp', 'ann', 'ni2', 'rev2', 'cogs']]

xj = xj.rename(columns={'经营性现金流-现金流量净额': 'ocf', '投资性现金流-现金流量净额': 'icf'})
xj = xj[['code', 'rp', 'ann', 'ocf', 'icf']]

df = yj.merge(zc.drop(columns=['ann']), on=['code', 'rp'], how='outer') \
       .merge(lr.drop(columns=['ann']), on=['code', 'rp'], how='outer') \
       .merge(xj.drop(columns=['ann']), on=['code', 'rp'], how='outer')
df['rev'] = df['rev'].fillna(df['rev2'])
df['ni'] = df['ni'].fillna(df['ni2'])
df = df.drop(columns=['rev2', 'ni2']).dropna(subset=['ann']).sort_values(['code', 'rp'])

# 累计 → 单季
g = df.groupby('code', sort=False)
for c in ['rev', 'ni', 'cogs', 'ocf', 'icf']:
    prev = g[c].shift(1)
    prev_y = g['rp'].shift(1).dt.year
    df[c + '_q'] = np.where(prev_y == df['rp'].dt.year, df[c] - prev, df[c])

for c in ['rev', 'ni', 'cogs', 'ocf', 'icf']:
    df[c + '_ttm'] = df.groupby('code', sort=False)[c + '_q'].transform(
        lambda x: x.rolling(4, min_periods=4).sum())

# 质量
EPS0 = 1e-9
df['roe_ttm'] = df['ni_ttm'] / df['eq'].abs().clip(lower=1)
df['roa_ttm'] = df['ni_ttm'] / df['ta'].clip(lower=1)
df['gpa'] = (df['rev_ttm'] - df['cogs_ttm']) / df['ta'].clip(lower=1)
df['gross_margin'] = df['gm'] / 100.0
df['ocf_ni'] = np.where(df['ni_ttm'] > 0, df['ocf_ttm'] / df['ni_ttm'].clip(lower=1), np.nan)
df['accrual'] = (df['ni_ttm'] - df['ocf_ttm']) / df['ta'].clip(lower=1)
df['debt_ratio'] = df['debt_ratio'] / 100.0
df['asset_turn'] = df['rev_ttm'] / df['ta'].clip(lower=1)

df = df.sort_values(['code', 'rp'])
g = df.groupby('code', sort=False)
df['roe_stab'] = g['roe_ttm'].transform(lambda x: x.rolling(12, min_periods=8).std())
df['roe_trend'] = g['roe_ttm'].transform(lambda x: x - x.shift(4))
df['gm_trend'] = g['gross_margin'].transform(lambda x: x - x.shift(4))
df['rev_yoy'] = df['rev_yoy_rpt'] / 100.0
df['profit_yoy'] = df['profit_yoy_rpt'] / 100.0
for c in ['recv', 'inv']:
    yy = g[c].transform(lambda x: x / x.shift(4).where(x.shift(4) > 0) - 1)
    ry = g['rev'].transform(lambda x: x / x.shift(4).where(x.shift(4) > 0) - 1)
    df[c + '_gap'] = yy - ry
df['ocf_yoy'] = g['ocf_ttm'].transform(lambda x: x / x.shift(4).where(x.shift(4) > 0) - 1)

# 估值分子（分母市值下一步合并）
df['ep'] = df['ni_ttm']
df['bp'] = df['eq']
df['sp'] = df['rev_ttm']
df['ocf_yield'] = df['ocf_ttm']
df['fcf_yield'] = df['ocf_ttm'] + df['icf_ttm']

FINCOLS = ['roe_ttm', 'roa_ttm', 'gpa', 'gross_margin', 'ocf_ni', 'accrual', 'debt_ratio',
           'asset_turn', 'roe_stab', 'roe_trend', 'gm_trend', 'rev_yoy', 'profit_yoy',
           'recv_gap', 'inv_gap', 'ocf_yoy', 'ep', 'bp', 'sp', 'ocf_yield', 'fcf_yield']

for c, lo, hi in [('roe_ttm', -3, 3), ('roa_ttm', -1, 1), ('gpa', -2, 2), ('ocf_ni', -5, 10),
                  ('accrual', -1, 1), ('asset_turn', 0, 5), ('roe_stab', 0, 2),
                  ('roe_trend', -2, 2), ('gm_trend', -1, 1), ('rev_yoy', -2, 3),
                  ('profit_yoy', -3, 3), ('ocf_yoy', -3, 3), ('recv_gap', -3, 3),
                  ('inv_gap', -3, 3), ('gross_margin', -1, 1), ('debt_ratio', 0, 2)]:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(lo, hi)

df = df[['code', 'rp', 'ann', 'sw_ind', 'ni', 'eps'] + FINCOLS]
print('财报因子 %d 行 / %d 只 / %d 期' % (len(df), df.code.nunique(), df.rp.nunique()), flush=True)

# ---------------- 市值 ----------------
mk = []
for f in sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet')):
    d = pd.read_parquet(f, columns=['date', 'symbol', 'amount', 'volume', 'turn'])
    px = d['amount'] / (d['volume'] * 100.0)                     # 真实成交均价
    fcap = d['amount'] * 100.0 / d['turn'].where(d['turn'] > 0)   # 流通市值
    mk.append(pd.DataFrame({'date': d['date'], 'code': d['symbol'].astype(str).str.zfill(6),
                            'px': px, 'fcap': fcap}))
mk = pd.concat(mk, ignore_index=True)
mk = mk[(mk.px > 0) & (mk.fcap > 1e8)]
print('行情 %d 行 / %d 只' % (len(mk), mk.code.nunique()), flush=True)

# ---------------- PIT ----------------
df = df.sort_values(['code', 'ann', 'rp']).drop_duplicates(['code', 'ann'], keep='last')
out = []
mk = mk.sort_values(['code', 'date'])
mk['date'] = pd.to_datetime(mk['date']).astype('datetime64[ns]')
mk_by = {k: v for k, v in mk.groupby('code', sort=False)}
df['ann'] = pd.to_datetime(df['ann']).astype('datetime64[ns]')
for code, gg in df.groupby('code', sort=False):
    m = mk_by.get(code)
    if m is None: continue
    out.append(pd.merge_asof(m, gg.drop(columns=['code']).sort_values('ann'),
                             left_on='date', right_on='ann', direction='backward'))
fp = pd.concat(out, ignore_index=True)
print('PIT 展开 %d 行' % len(fp), flush=True)

# 总市值：总股本 = 累计净利润 ÷ 累计 EPS，退化时用流通市值
shares = fp['ni'] / fp['eps'].where(fp['eps'].abs() > 1e-4)
mcap = shares * fp['px']
ok = mcap.notna() & (mcap > 1e7) & (mcap < 1e13) & (mcap > fp['fcap'] * 0.5)
fp['mcap'] = np.where(ok, mcap, fp['fcap'])
print('总市值反推成功率 %.1f%%' % (ok.mean() * 100), flush=True)

for c in ['ep', 'bp', 'sp', 'ocf_yield', 'fcf_yield']:
    fp[c] = fp[c] / fp['mcap']
    fp[c] = fp[c].replace([np.inf, -np.inf], np.nan).clip(-1, 1)

fp = fp.rename(columns={'code': 'symbol'})
sel = ['date', 'symbol'] + FINCOLS
out_df = fp[sel].copy()
for c in FINCOLS:
    out_df[c] = out_df[c].astype(np.float32)
out_df.to_parquet(OUT, index=False)
print('\n已存 %s（%d 行）' % (OUT, len(fp)))
for c in FINCOLS:
    print('  %-12s 非空 %.1f%%' % (c, fp[c].notna().mean() * 100))

sw = df[df.sw_ind.notna()][['code', 'ann', 'sw_ind']].copy()
sw['year'] = sw['ann'].dt.year
sw = sw.sort_values('ann').drop_duplicates(['code', 'year'], keep='last')
sw[['code', 'year', 'sw_ind']].to_parquet(f'{ROOT}/data/big/sw_industry.parquet', index=False)
print('\n申万二级行业：%d 行 / %d 个行业' % (len(sw), sw.sw_ind.nunique()))
