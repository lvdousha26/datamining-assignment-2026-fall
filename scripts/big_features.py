# -*- coding: utf-8 -*-
"""全市场面板 + 因子（规模：5000+ 只 × 2015-2026，42 个因子）
输出：
  data/big/panel_big.parquet   长表（训练段，含 industry / is_st）
  data/big/arr.npz             X[T,N,F] / Y[T,N] / Y2[T,N] / M[T,N] / IND[T,N]
  data/big/lockbox_labels.npz  封存段的 (ti,ni,y2)，主数组里该段 Y/Y2 已置 NaN
注意：
  价格前复权；收益率用 qfq 计算（连续，无除权跳空）
  y = 未来 5 日收益，截面去均值；y_rk = 截面百分位
  ST：baostock 时变 isST 日历，ST 当日样本剔除
  行业：ind_roll.parquet（每年只用上一年及以前的收益聚类），point-in-time 无前视
  lockbox：LOCK_START 之后只保留特征 X，标签物理隔离到单独文件
"""
import os, glob, numpy as np, pandas as pd

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
RAW = f'{ROOT}/data/big/raw'
HFQ = f'{ROOT}/data/big/raw_hfq'
OUT = f'{ROOT}/data/big'
os.makedirs(OUT, exist_ok=True)

# 数据源：默认后复权（hfq）。东财前复权是「减法复权」，高分红老股会被压到 0 甚至负数
# （茅台 qfq 最小价 −143.50），导致收益率方向反转。hfq 无此问题。
SRC = os.environ.get('SRC', 'hfq')
_hfq_files = sorted(glob.glob(f'{HFQ}/*.parquet'))
if SRC == 'hfq' and len(_hfq_files) > 1000:
    files = _hfq_files
    print('数据源 hfq：%d 只（后复权）' % len(files), flush=True)
else:
    files = sorted(glob.glob(f'{RAW}/shard_*.parquet'))
    print('数据源 qfq：%d 个分片（前复权，仅回退用）' % len(files), flush=True)
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'pctChg', 'turn']:
    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
# 停牌/异常行 close=0：会让未来收益标签产生 inf，并在下面的截面去均值中扩散到当日全市场
# （历史教训：70 只停牌股的 13505 行 → 扩散成 23 万条 ±inf 脏标签、82 个交易日）
df.loc[df['close'] <= 0, 'close'] = np.nan
df['symbol'] = df['symbol'].astype(str).str.zfill(6)
df = df[df['date'] >= '2015-01-01'].dropna(subset=['close']).sort_values(['symbol', 'date']).reset_index(drop=True)
print('原始 %d 行 | %d 只 | %s ~ %s' % (len(df), df['symbol'].nunique(),
      df['date'].min().date(), df['date'].max().date()), flush=True)

# ---- ST 剔除（baostock 时变 isST）----
st_file = ('st_calendar_wide.parquet' if os.environ.get('ST_WIDE') == '1'
           else 'st_calendar.parquet')
st = pd.read_parquet(f'{OUT}/{st_file}').drop(columns=['source'], errors='ignore')
st['is_st'] = np.int8(1)
print('ST 口径:', st_file, flush=True)
df = df.merge(st, on=['date', 'symbol'], how='left', sort=False)
df['is_st'] = df['is_st'].fillna(0).astype(np.int8)
print('ST 样本 %d 行 (%.2f%%) | 涉及 %d 只' %
      (df.is_st.sum(), df.is_st.mean() * 100, df.loc[df.is_st == 1, 'symbol'].nunique()), flush=True)

# ---- point-in-time 行业 ----
# 首选申万二级（业务分类，跨年稳定、可解释）；缺失时回退到 ind_roll 的统计聚类 31 组。
# 按「该年 7/1」向后 merge_asof 铺到交易日，不用未来年份，严格 PIT。
swf = f'{OUT}/sw_industry.parquet'
if os.path.exists(swf):
    sw = pd.read_parquet(swf)
    sw['ann'] = pd.to_datetime(dict(year=sw['year'], month=7, day=1)).astype('datetime64[ns]')
    sw['code'] = sw['code'].astype(str).str.zfill(6)
    vc = sw['sw_ind'].value_counts(); keep = set(vc[vc >= 30].index)
    sw['ind'] = sw['sw_ind'].where(sw['sw_ind'].isin(keep), '其他')
    codes = {k: i for i, k in enumerate(sorted(sw['ind'].unique()))}
    sw['ind_id'] = sw['ind'].map(codes).astype(np.int8)
    swg = {k: v.sort_values('ann')[['ann', 'ind_id']] for k, v in sw.groupby('code')}
    ind_col = np.full(len(df), -1, dtype=np.int8)
    for code, gg in df.groupby('symbol', sort=False):
        m = swg.get(code)
        if m is None: continue
        _l = gg[['date']].astype({'date': 'datetime64[ns]'}).sort_values('date')
        a = pd.merge_asof(_l, m, left_on='date', right_on='ann', direction='backward')
        ind_col[gg.index.values] = a['ind_id'].fillna(-1).astype(np.int8).values
    df['industry'] = ind_col
    src = 'sw_industry(申万二级)'
else:
    ind = pd.read_parquet(f'{OUT}/ind_roll.parquet')
    ind['symbol'] = ind['symbol'].astype(str).str.zfill(6)
    df['year'] = df['date'].dt.year
    df = df.merge(ind, on=['year', 'symbol'], how='left', sort=False)
    df['industry'] = df['cluster'].fillna(-1).astype(np.int8)
    df = df.drop(columns=['cluster'])
    src = 'ind_roll(聚类31组)'
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
print('行业来源 %s | 覆盖 %.1f%% | %d 个分组（含未分类 -1）' %
      (src, (df.industry >= 0).mean() * 100, df.industry.nunique()), flush=True)

g = df.groupby('symbol', sort=False)
c, o, h, l, v, amt, tur = df['close'], df['open'], df['high'], df['low'], df['volume'], df['amount'], df['turn']
pc = g['close'].shift(1)
ret1 = c / pc - 1
df['ret1'] = ret1

# ---- 复权断层识别（用涨跌停制度做硬约束反查）----
# A 股日涨跌幅有法定上限：主板 10%、创业板 2020-08-24 后 20%、科创板 20%、北交所 30%。
# 超出上限 +2pp、且不属于「上市前 5 日 / 停牌超 10 日复牌」的，只可能是复权处理出错
# （拆股、送转、分红）。这类价格是假的，任何跨断层的收益与价格类因子都必须作废。
_sym = df['symbol']
_lim = np.full(len(df), 0.10, dtype=np.float64)
_lim[_sym.str.startswith('688').values] = 0.20
_cyb = _sym.str.startswith('30').values
if _cyb.any():
    _lim[_cyb] = np.where(df.loc[_cyb, 'date'].values >= np.datetime64('2020-08-24'), 0.20, 0.10)
_lim[_sym.str[0].isin(['8', '4']).values] = 0.30
_age = df.groupby('symbol', sort=False).cumcount().values
_gap = (df['date'] - df.groupby('symbol', sort=False)['date'].shift(1)).dt.days.values
_gap = np.nan_to_num(_gap.astype(float), nan=0.0)
df['brk'] = (np.abs(df['ret1'].values) > _lim + 0.02) & (_age >= 5) & (_gap <= 10)
print('复权断层 %d 处（%.3f%%）| 涉及 %d 只' %
      (int(df['brk'].sum()), df['brk'].mean() * 100,
       df.loc[df['brk'], 'symbol'].nunique()), flush=True)
# 断层当日的「收益」是假的，置 NaN：滚动统计（std/skew/illiq/pvcorr）会用 min_periods 跳过它，
# 不会被 −99% 这类假值带偏。价格比值类因子（ret20/ret60）仍会跨断层，见报告「未闭合项」。
df.loc[df['brk'], 'ret1'] = np.nan

print('计算因子...', flush=True)
# 动量/反转
for w in [5, 10, 20, 60, 120, 250]:
    df[f'ret{w}'] = g['close'].transform(lambda x, w=w: x / x.shift(w) - 1)
# 波动
for w in [20, 60]:
    df[f'std{w}'] = g['ret1'].transform(lambda x, w=w: x.rolling(w, min_periods=max(5, w // 3)).std())
df['skew20'] = g['ret1'].transform(lambda x: x.rolling(20, min_periods=8).skew())
df['kurt20'] = g['ret1'].transform(lambda x: x.rolling(20, min_periods=8).kurt())
df['maxret20'] = g['ret1'].transform(lambda x: x.rolling(20, min_periods=8).max())
df['down20'] = g['ret1'].transform(lambda x: x.where(x < 0, 0).rolling(20, min_periods=8).std())
# 换手
for w in [5, 20, 60]:
    df[f'turn{w}'] = g['turn'].transform(lambda x, w=w: x.rolling(w, min_periods=max(3, w // 3)).mean())
df['turn_ratio'] = df['turn5'] / (df['turn60'] + 1e-6)
df['turn_std20'] = g['turn'].transform(lambda x: x.rolling(20, min_periods=6).std())
# 振幅
df['amp20'] = g['amplitude'].transform(lambda x: x.rolling(20, min_periods=6).mean())
# 日内/隔夜
df['intra'] = c / o - 1
df['overn'] = o / pc - 1
for w in [5, 20]:
    df[f'intra{w}'] = g['intra'].transform(lambda x, w=w: x.rolling(w, min_periods=max(3, w // 3)).mean())
    df[f'overn{w}'] = g['overn'].transform(lambda x, w=w: x.rolling(w, min_periods=max(3, w // 3)).mean())
# 非流动性（Amihud）
df['illiq'] = df['ret1'].abs() / (df['amount'] + 1e-6) * 1e8
df['illiq20'] = g['illiq'].transform(lambda x: x.rolling(20, min_periods=6).mean())
# 量比
df['vol5'] = g['volume'].transform(lambda x: x.rolling(5, min_periods=2).mean())
df['vol60'] = g['volume'].transform(lambda x: x.rolling(60, min_periods=20).mean())
df['volratio'] = df['vol5'] / (df['vol60'] + 1)
# 位置/均线偏离
mn250 = g['close'].transform(lambda x: x.rolling(250, min_periods=60).min())
mx250 = g['close'].transform(lambda x: x.rolling(250, min_periods=60).max())
df['pos250'] = (c - mn250) / (mx250 - mn250 + 1e-6)
df['dev20'] = c / g['close'].transform(lambda x: x.rolling(20, min_periods=6).mean()) - 1
df['dev60'] = c / g['close'].transform(lambda x: x.rolling(60, min_periods=20).mean()) - 1
# 规模代理/风险调整动量
df['size'] = np.log(g['amount'].transform(lambda x: x.rolling(20, min_periods=6).mean()) + 1)
df['momq60'] = df['ret60'] / (df['std60'] + 1e-6)

# ---- 量价专项（12 个）----
# 成交量单位是「手」，VWAP = 成交额 / (成交量*100)
vwap = df['amount'] / (df['volume'] * 100 + 1e-6)
rng = (h - l).replace(0, np.nan)
df['clv'] = ((c - l) / rng).fillna(0)                      # 收盘在当日区间的位置
df['ushadow'] = ((h - np.maximum(o, c)) / rng).fillna(0)   # 上影线
df['lshadow'] = ((np.minimum(o, c) - l) / rng).fillna(0)   # 下影线
df['vwap_dev'] = c / vwap.replace(0, np.nan) - 1           # 收盘相对 VWAP 偏离
dvol = g['volume'].transform(lambda x: x.pct_change())
df['dv'] = dvol
df['pvcorr20'] = df.groupby('symbol', sort=False).apply(
    lambda x: x['ret1'].rolling(20, min_periods=10).corr(x['dv']), include_groups=False).reset_index(level=0, drop=True)
df['pvcorr60'] = df.groupby('symbol', sort=False).apply(
    lambda x: x['ret1'].rolling(60, min_periods=25).corr(x['dv']), include_groups=False).reset_index(level=0, drop=True)
amt20 = g['amount'].transform(lambda x: x.rolling(20, min_periods=6).mean())
amt250 = g['amount'].transform(lambda x: x.rolling(250, min_periods=60).mean())
df['amt_ratio'] = amt20 / (amt250 + 1)
turn60s = g['turn'].transform(lambda x: x.rolling(60, min_periods=20).std())
df['turn_z'] = (tur - g['turn'].transform(lambda x: x.rolling(60, min_periods=20).mean())) / (turn60s + 1e-6)
amp250 = g['amplitude'].transform(lambda x: x.rolling(250, min_periods=60).mean())
df['range_ratio'] = df['amp20'] / (amp250 + 1e-6)          # 振幅收缩/扩张
df['mf20'] = (g.apply(lambda x: (x['ret1'] * x['amount']).rolling(20, min_periods=6).sum(), include_groups=False)
              .reset_index(level=0, drop=True) / (amt20 * 20 + 1))   # 资金加权收益
df['wvma20'] = c / (g.apply(lambda x: (x['close'] * x['amount']).rolling(20, min_periods=6).sum(), include_groups=False)
                    .reset_index(level=0, drop=True) / (amt20 * 20 + 1)) - 1
illiq250 = g['illiq'].transform(lambda x: x.rolling(250, min_periods=60).mean())
df['illiq_ratio'] = df['illiq20'] / (illiq250 + 1e-12)

# ---- 价格比值/回看窗口类因子的断层感知 ----
# ret{w}、pos250、dev20/60、momq60、wvma20 都是「当前价 ÷ 窗口内某个价格统计量」，
# 只要回看窗口里有一天是断层，分子分母就不在同一复权尺度上，比值完全没有经济含义。
# 表现：断层后 20 日的 ret20 均值 +0.750（正常 +0.149），被系统性打上 0.7σ 的「假动量」。
# 处理：窗口跨断层 → 置 NaN。
_price_lb = [('ret5', 5), ('ret10', 10), ('ret20', 20), ('ret60', 60),
             ('ret120', 120), ('ret250', 250), ('dev20', 20), ('dev60', 60),
             ('pos250', 250), ('momq60', 60), ('wvma20', 20)]
_bcs = df.groupby('symbol', sort=False)['brk'].cumsum()
_lb_cache, _killed = {}, {}
for _col, _L in _price_lb:
    if _L not in _lb_cache:
        _prev = _bcs.groupby(df['symbol'], sort=False).shift(_L).fillna(0)
        _lb_cache[_L] = ((_bcs - _prev) > 0).values      # t-L+1 .. t 含断层
    _hit = _lb_cache[_L] & np.isfinite(df[_col].values)
    df.loc[_hit, _col] = np.nan
    _killed[_col] = int(_hit.sum())
print('价格类因子跨断层作废：' + ' | '.join('%s %d' % (k, v) for k, v in _killed.items()), flush=True)

# 标签：未来 5 日收益。先清掉 inf/nan，再做截面去均值——
# 否则单个 inf 会污染 s.mean()，把 inf 扩散给当日全部股票。
df['fwd5'] = g['close'].transform(lambda x: x.shift(-5) / x - 1).replace([np.inf, -np.inf], np.nan)
# 可执行口径：次日收盘买入，再持有 5 日（信号只能用 t 日及之前信息）
df['fwd5_lag'] = g['close'].transform(lambda x: x.shift(-6) / x.shift(-1) - 1).replace([np.inf, -np.inf], np.nan)


# 跨复权断层的收益一律作废：断层两侧价格不在同一复权尺度上，比值没有经济含义。
#   fwd5     = close[t+5] / close[t]     → 跨度 t+1..t+5
#   fwd5_lag = close[t+6] / close[t+1]   → 跨度 t+2..t+6
def _cross_brk(k_from, k_to):
    out = np.zeros(len(df), dtype=bool)
    gb = df.groupby('symbol', sort=False)['brk']
    for k in range(k_from, k_to + 1):
        out |= gb.shift(-k).fillna(False).astype(bool).values
    return out


_c5 = _cross_brk(1, 5)
_c6 = _cross_brk(2, 6)
df.loc[_c5, 'fwd5'] = np.nan
df.loc[_c6, 'fwd5_lag'] = np.nan
print('因跨断层作废：fwd5 %d 行 | fwd5_lag %d 行' % (int(_c5.sum()), int(_c6.sum())), flush=True)

df['y'] = df.groupby('date')['fwd5'].transform(lambda s: s - s.mean())
df['y_rk'] = df.groupby('date')['fwd5'].transform(lambda s: s.rank(pct=True))
df['y2'] = df.groupby('date')['fwd5_lag'].transform(lambda s: s - s.mean())
df['y2_rk'] = df.groupby('date')['fwd5_lag'].transform(lambda s: s.rank(pct=True))

FC = ['ret1', 'ret5', 'ret10', 'ret20', 'ret60', 'ret120', 'ret250',
      'std20', 'std60', 'skew20', 'kurt20', 'maxret20', 'down20',
      'turn5', 'turn20', 'turn60', 'turn_ratio', 'turn_std20', 'amp20',
      'intra5', 'intra20', 'overn5', 'overn20', 'illiq20',
      'volratio', 'pos250', 'dev20', 'dev60', 'size', 'momq60',
      'clv', 'ushadow', 'lshadow', 'vwap_dev', 'pvcorr20', 'pvcorr60',
      'amt_ratio', 'turn_z', 'range_ratio', 'mf20', 'wvma20', 'illiq_ratio']
print('因子 %d 个' % len(FC), flush=True)

# 清洗：去极值 + 截面 z-score（逐日）—— 全部用 C 层 groupby，避免 2600 次 lambda
print('截面标准化...', flush=True)
for f in FC:
    med = df.groupby('date')[f].transform('median')
    mad = (df[f] - med).abs().groupby(df['date']).transform('median') * 1.4826 + 1e-12
    df[f] = ((df[f].clip(med - 5 * mad, med + 5 * mad) - med) / mad).clip(-4, 4).astype('float32')

# 过滤：成交额下限、上市满 250 日、非 ST、标签与因子有限
df['ok'] = ((df['amount'] > 5e6) & df['ret250'].notna() & (df['is_st'] == 0)
            & np.isfinite(df['y']) & np.isfinite(df['y2'])
            & np.isfinite(df[FC]).all(axis=1))
sub = df[df['ok']].reset_index(drop=True)
LOCK_START = pd.Timestamp('2025-07-01')
lock = sub['date'] >= LOCK_START
trs = sub[~lock]
print('有效样本 %d 行 | %d 日 | %d 只' % (len(sub), sub['date'].nunique(), sub['symbol'].nunique()), flush=True)
print('其中训练可用 %d 行 | 封存(lockbox) %d 行 / %d 日' %
      (len(trs), lock.sum(), sub.loc[lock, 'date'].nunique()), flush=True)
trs[['date', 'symbol', 'y', 'y_rk', 'fwd5', 'fwd5_lag', 'y2', 'y2_rk', 'industry'] + FC].to_parquet(
    f'{OUT}/panel_big.parquet', index=False)

# ---- 三维数组（供 Transformer 切片）----
# 封存段保留特征 X（评估要用），但 Y/Y2 只在训练段填值，标签单独物理隔离。
dates = np.sort(sub['date'].unique()); syms = np.sort(sub['symbol'].unique())
di = {d: i for i, d in enumerate(dates)}; si = {s: i for i, s in enumerate(syms)}
T, N, F = len(dates), len(syms), len(FC)
X = np.zeros((T, N, F), dtype=np.float32); Y = np.full((T, N), np.nan, dtype=np.float32)
Y2 = np.full((T, N), np.nan, dtype=np.float32)
IND = np.full((T, N), -1, dtype=np.int8)
M = np.zeros((T, N), dtype=bool)
ti = sub['date'].map(di).values; ni = sub['symbol'].map(si).values
X[ti, ni, :] = sub[FC].values.astype(np.float32)
IND[ti, ni] = sub['industry'].values.astype(np.int8)
M[ti, ni] = True
t1 = trs['date'].map(di).values; n1 = trs['symbol'].map(si).values
Y[t1, n1] = trs['y'].values.astype(np.float32)
Y2[t1, n1] = trs['y2'].values.astype(np.float32)
np.savez(f'{OUT}/arr.npz', X=X, Y=Y, Y2=Y2, M=M, IND=IND,
         dates=dates.astype('datetime64[D]').astype(str), symbols=syms,
         feats=np.array(FC), lock_start=str(LOCK_START.date()))

lks = sub[lock]
np.savez(f'{OUT}/lockbox_labels.npz',
         ti=lks['date'].map(di).values.astype(np.int32),
         ni=lks['symbol'].map(si).values.astype(np.int32),
         y=lks['y'].values.astype(np.float32),
         y2=lks['y2'].values.astype(np.float32),
         dates=dates.astype('datetime64[D]').astype(str),
         lock_start=str(LOCK_START.date()))
print('数组 X%s | 训练标签 %.0f%% | 封存标签已隔离到 lockbox_labels.npz' %
      (X.shape, np.isfinite(Y2[M]).mean() * 100), flush=True)
print('完成', flush=True)
