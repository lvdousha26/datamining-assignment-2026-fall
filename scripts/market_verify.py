# -*- coding: utf-8 -*-
"""市场基准三源核对：全市场等权 vs 沪深300 vs 中证全指

目的：
  1) 用日频 pctChg（东财口径，已含复权）重建「全市场等权日收益」，
     这是 Y2（截面去均值）真正的基准；
  2) 与沪深300、中证全指逐年对照，确认等权序列没有口径错误；
  3) 输出可复用的日收益宽表 data/big/daily_ret.npz（供绝对收益回测使用）。
"""
import numpy as np, pandas as pd, glob

ROOT = 'D:/quant'
# ---------- ① 个股日收益（pctChg 为东财官方涨跌幅，已复权，不受 close 断层影响）----------
px = []
_HFQ = f'{ROOT}/data/big/raw_hfq'
_hf = sorted(glob.glob(f'{_HFQ}/*.parquet'))
_src = _hf if len(_hf) > 1000 else sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet'))
_cols = ['date', 'symbol', 'pctChg', 'close', 'amount']
for f in _src:
    px.append(pd.read_parquet(f, columns=_cols))
px = pd.concat(px, ignore_index=True)
print('收益矩阵数据源：%s（%d 个文件）' % ('hfq 后复权' if len(_hf) > 1000 else 'qfq 前复权', len(_src)), flush=True)
px['date'] = pd.to_datetime(px['date'])
px['symbol'] = px['symbol'].astype(str).str.zfill(6)
px.loc[px['close'] <= 0, 'pctChg'] = np.nan          # 停牌：close=0 时涨跌幅无意义
px['r'] = px['pctChg'] / 100.0

# ---- 复权断层识别（涨跌停硬约束反查）----
# 超出法定涨跌幅上限 +2pp、且非「上市前 5 日 / 停牌超 10 日复牌」的日收益，
# 是拆股/送转/分红导致的复权错误，不是真实涨跌。这类收益必须置 NaN 而非 0：
# 置 0 等于告诉组合「这天没涨没跌」，会把假暴跌的损失藏起来、系统性低估风险。
px = px.sort_values(['symbol', 'date']).reset_index(drop=True)
_sym = px['symbol']
_lim = np.full(len(px), 0.10)
_lim[_sym.str.startswith('688').values] = 0.20
_cyb = _sym.str.startswith('30').values
if _cyb.any():
    _lim[_cyb] = np.where(px.loc[_cyb, 'date'].values >= np.datetime64('2020-08-24'), 0.20, 0.10)
_lim[_sym.str[0].isin(['8', '4']).values] = 0.30
_age = px.groupby('symbol', sort=False).cumcount().values
_gap = (px['date'] - px.groupby('symbol', sort=False)['date'].shift(1)).dt.days.values
_gap = np.nan_to_num(_gap.astype(float), nan=0.0)
brk = (np.abs(px['r'].values) > _lim + 0.02) & (_age >= 5) & (_gap <= 10)
print('剔除复权断层 %d 处（%.3f%%）' % (int(brk.sum()), brk.mean() * 100), flush=True)

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
dates = pd.to_datetime(z['dates']); T, N = z['Y2'].shape
syms = [str(s).zfill(6) for s in z['symbols']]

ti = pd.Index(dates).get_indexer(pd.DatetimeIndex(px['date']))
si = pd.Index(syms).get_indexer(px['symbol'])
ok = (ti >= 0) & (si >= 0) & np.isfinite(px['r'].values) & (px['r'].abs().values < 0.5) & ~brk
R = np.full((T, N), np.nan, np.float32)
R[ti[ok], si[ok]] = px['r'].values[ok].astype(np.float32)
print('日收益矩阵 %.0f%% 非空 | 股票 %d | 日 %d' % (np.isfinite(R).mean() * 100, N, T), flush=True)
_w = R[np.isfinite(R)]
print('  区间 [%.1f%%, %.1f%%] | |r|>25%% 占 %.4f%%' % (_w.min() * 100, _w.max() * 100,
      (np.abs(_w) > 0.25).mean() * 100), flush=True)

# ---------- ② 全市场等权（每日再平衡）----------
ew = pd.Series(np.nanmean(np.where(np.isfinite(R), R, np.nan), axis=1), index=dates)
n_stk = pd.Series(np.isfinite(R).sum(1), index=dates)

# ---------- ③ 指数 ----------
def idx_close(code, name):
    d = pd.read_pickle(f'{ROOT}/data/big/idx_{code}.pkl') if False else None
    return None

IDX = {}
import os
if os.path.exists(f'{ROOT}/data/big/idx_daily.pkl'):
    IDX = pd.read_pickle(f'{ROOT}/data/big/idx_daily.pkl')
    print('  指数数据：读取缓存 idx_daily.pkl')
import akshare as ak
for code, nm in ([] if IDX else [('000300', '沪深300'), ('000985', '中证全指')]):
    try:
        d = ak.index_zh_a_hist(symbol=code, period='daily',
                               start_date='20150101', end_date='20261031')
        d['日期'] = pd.to_datetime(d['日期'])
        IDX[nm] = d.set_index('日期')['收盘'].sort_index()
        print('  %s %d 行 %s~%s' % (nm, len(d), d['日期'].min().date(), d['日期'].max().date()), flush=True)
    except Exception as e:
        print('  %s FAIL %s' % (nm, str(e)[:60]), flush=True)
pd.to_pickle(IDX, f'{ROOT}/data/big/idx_daily.pkl')

# ---------- ④ 逐年对照 ----------
def yr_ret(s):
    s = s.dropna()
    return s.groupby(s.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)

tab = pd.DataFrame({'全市场等权': yr_ret(ew)})
for nm, s in IDX.items():
    tab[nm] = yr_ret(s.pct_change())
# 等权相对指数的「规模差」：等权 - 沪深300 ≈ 小盘超额
tab['等权-300'] = tab['全市场等权'] - tab['沪深300']
print('\n=== 逐年收益（%，日频复利）===')
print(tab.round(2).to_string())
print('\n=== 全期 ===')
for c in tab.columns[:3]:
    s = {'全市场等权': ew, '沪深300': IDX['沪深300'].pct_change(), '中证全指': IDX['中证全指'].pct_change()}[c].dropna()
    ann = ((1 + s).prod() ** (244 / len(s)) - 1) * 100
    vol = s.std() * 244 ** .5
    print('  %-8s 年化 %+6.2f%% | 波动 %5.2f%% | 夏普 %.2f | 日数 %d'
          % (c, ann, vol * 100, (ann / 100 - 0.02) / vol, len(s)))

# 相关性
m = pd.DataFrame({'ew': ew, 'hs300': IDX['沪深300'].pct_change(),
                  'csi_all': IDX['中证全指'].pct_change()}).dropna()
print('\n日收益相关：等权-300 %.3f | 等权-全指 %.3f | 300-全指 %.3f'
      % (m.ew.corr(m.hs300), m.ew.corr(m.csi_all), m.hs300.corr(m.csi_all)))

# ---------- ⑤ 落盘 ----------
np.savez_compressed(f'{ROOT}/data/big/daily_ret.npz', R=R, dates=dates.values.astype('datetime64[D]'),
                    symbols=np.array(syms))
print('\n已存 data/big/daily_ret.npz (R %s) 与 data/big/idx_daily.pkl' % (R.shape,))
