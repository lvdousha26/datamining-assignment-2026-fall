# -*- coding: utf-8 -*-
"""B 级策略能不能融券做空：D1 组合的做空经济性

两种做空方式都算：
  A 纯做空 D1（全空，有市场 beta 暴露）
  B 做空 D1 + 做多市场等权（市场中性，赚的是 D1 的超额）

成本：融券费率 6/8/10/12%/年（A 股实际 8-10%），交易 20bp 单边，换手 6 倍/年
"""
import glob
import numpy as np, pandas as pd

ROOT = 'D:/quant'
STEP = 5

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
M, dates = z['M'], pd.to_datetime(z['dates'])
T, N = M.shape
syms = [str(s).zfill(6) for s in z['symbols']]
sidx = {s: j for j, s in enumerate(syms)}

# 原始 5 日收益（绝对）
px = []
for f in sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet')):
    d = pd.read_parquet(f, columns=['date', 'symbol', 'close'])
    d['symbol'] = d['symbol'].astype(str).str.zfill(6)
    px.append(d)
px = pd.concat(px, ignore_index=True).sort_values(['symbol', 'date'])
# 停牌股 close=0 → 必须置 NaN，否则 fwd5 = 0/x-1 = -100% 的假收益（历史踩过的坑）
px.loc[px['close'] <= 0, 'close'] = np.nan
px['fwd5'] = px.groupby('symbol')['close'].transform(lambda s: s.shift(-5) / s - 1)
px = px.replace([np.inf, -np.inf], np.nan)
F = np.full((T, N), np.nan, np.float32)
ti = pd.Index(dates).get_indexer(pd.DatetimeIndex(px['date']))
si = pd.Index(syms).get_indexer(px['symbol'])
f5 = px['fwd5'].values
ok = (ti >= 0) & (si >= 0) & np.isfinite(f5)
f5 = np.clip(f5, -0.5, 0.5)      # 截尾：前复权断层会产生 -238 / +383 这种假收益
F[ti[ok], si[ok]] = f5[ok]
print('5日收益矩阵就绪，非空 %.1f%%' % (np.isfinite(F).mean() * 100), flush=True)

p = np.load(f'{ROOT}/data/big/preds.npz')
tt = np.concatenate([p['tt_me'], p['tt_te']])
ii = np.concatenate([p['ii_me'], p['ii_te']])
pr = np.concatenate([p['p_lgb_me'], p['p_lgb_te']])
seg = {}
for t, s, v in zip(tt, ii, pr):
    seg.setdefault(t, []).append((s, v))
seg = {k: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for k, v in seg.items()}
tl = sorted(seg)

rows = []
prev_d1 = None
for t in tl[::STEP]:
    s, v = seg[t]
    f = F[t, s]
    ok2 = np.isfinite(f)
    s, v, f = s[ok2], v[ok2], f[ok2]
    if len(s) < 300: continue
    n = max(int(len(s) * 0.10), 10)
    bot = s[np.argsort(v)[:n]]
    fb = F[t, bot]; fb = fb[np.isfinite(fb)]
    if len(fb) < 10: continue
    r_d1 = float(fb.mean())
    m = M[t] & np.isfinite(F[t])
    r_mkt = float(F[t][m].mean())
    rows.append((t, r_d1, r_mkt, len(fb)))
    prev_d1 = set(bot)

d = pd.DataFrame(rows, columns=['t', 'r_d1', 'r_mkt', 'n'])
d['ex'] = d.r_d1 - d.r_mkt
d['yr'] = pd.DatetimeIndex(dates[d.t]).year
PPY = 252 / STEP
print('对照段 %s ~ %s | %d 期 | D1 平均 %d 只' %
      (dates[d.t.min()].date(), dates[d.t.max()].date(), len(d), d.n.mean()), flush=True)


def ann(x):
    return ((1 + x).prod() ** (PPY / len(x)) - 1) * 100


print('\n=== ① D1 本身的表现（年化）===')
print('  D1 绝对收益      %+.2f%%' % ann(d.r_d1))
print('  市场等权         %+.2f%%' % ann(d.r_mkt))
print('  D1 超额（=−做空毛收益）%+.2f%%' % ann(d.ex))
print('  D1 波动 %.1f%% | 超额波动 %.1f%%' % (d.r_d1.std() * np.sqrt(PPY) * 100,
                                              d.ex.std() * np.sqrt(PPY) * 100))

print('\n=== ② 纯做空 D1（不对冲，全空）===')
gross = -d.r_d1
print('  毛收益 %+.2f%% / 波动 %.1f%% / 夏普 %.2f' %
      (ann(gross), gross.std() * np.sqrt(PPY) * 100,
       (ann(gross) / 100 - .02) / (gross.std() * np.sqrt(PPY))))

print('\n=== ③ 做空 D1 + 做多市场（市场中性）===')
gross_n = -d.ex
print('  毛收益 %+.2f%% / 波动 %.1f%%' % (ann(gross_n), gross_n.std() * np.sqrt(PPY) * 100))
print('  %-10s %10s %10s %10s' % ('融券费率', '6%', '8%', '10%'))
for tc in [0.0020]:
    print('  %-10s %+10.2f%% %+10.2f%% %+10.2f%%' %
          ('净收益', ann(gross_n) - 6 - tc * 2 * PPY, ann(gross_n) - 8 - tc * 2 * PPY,
           ann(gross_n) - 10 - tc * 2 * PPY))

print('\n=== ④ 分年：D1 绝对收益 vs 超额 ===')
by = d.groupby('yr').apply(lambda g: pd.Series({
    'D1绝对%': ann(g.r_d1), '市场%': ann(g.r_mkt), '超额%': ann(g.ex)}), include_groups=False)
print(by.round(1).to_string())

# 可融券比例（深市名单）
try:
    import akshare as ak
    mg = ak.stock_margin_underlying_info_szse(date=dates[max(tl)].strftime('%Y%m%d'))
    mg_set = set(mg['证券代码'].astype(str).str.zfill(6))
    last_t = max(tl)
    s, v = seg[last_t]
    n = max(int(len(s) * 0.10), 10)
    d1_last = [syms[k] for k in s[np.argsort(v)[:n]]]
    inpool = sum(1 for c in d1_last if c in mg_set)
    sz = sum(1 for c in d1_last if c[:2] in ('00', '30'))
    print('\n=== ⑤ 当期 D1 的可融券情况 ===')
    print('  D1 %d 只 | 深市 %d 只 | 深市名单内 %d 只 (%.0f%%)' %
          (len(d1_last), sz, inpool, inpool / max(sz, 1) * 100))
    print('  注：名单只代表"在标的池"，不代表券商有券可借；A股融券余额约占两融 2%')
except Exception as e:
    print('\n两融名单获取失败:', str(e)[:80])

by.round(2).to_csv(f'{ROOT}/reports/short_d1.csv')
print('\n已存 reports/short_d1.csv')
