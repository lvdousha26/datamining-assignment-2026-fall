# -*- coding: utf-8 -*-
"""质量+估值长线策略：容量与暴露

1. 复现最新一期选股（质量前 40% → 估值 top300）
2. 市值 / 流动性 / 行业暴露：与全市场对比
3. 容量估算：单票不超过 ADV 的 5% / 10% / 20% 三档
4. 冲击成本：平方根模型 cost = 0.1 × σ_d × sqrt(Q/ADV)
"""
import glob
import numpy as np, pandas as pd

ROOT = 'D:/quant'
TOP = 300

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
M, dates = z['M'], pd.to_datetime(z['dates'])
T, N = M.shape
IND = np.load(f'{ROOT}/data/big/IND_sw.npy')
names = pd.read_json(f'{ROOT}/data/big/IND_sw_names.json', typ='series')
syms = [str(s).zfill(6) for s in z['symbols']]

# 市值与成交额（原始行情）
mk = []
for f in sorted(glob.glob(f'{ROOT}/data/big/raw/shard_*.parquet')):
    d = pd.read_parquet(f, columns=['date', 'symbol', 'amount', 'turn'])
    d['symbol'] = d['symbol'].astype(str).str.zfill(6)
    mk.append(d)
mk = pd.concat(mk, ignore_index=True)
mk['date'] = pd.to_datetime(mk['date'])
mk['mcap'] = mk['amount'] * 100.0 / mk['turn'].where(mk['turn'] > 0)   # 流通市值
print('行情 %d 行 / %d 只' % (len(mk), mk.symbol.nunique()), flush=True)

# 最新交易日
t0 = T - 1
d0 = dates[t0]
day = mk[mk.date == d0].set_index('symbol')
adv20 = (mk[mk.date >= d0 - pd.Timedelta(days=40)]
         .groupby('symbol')['amount'].agg(['mean', 'median', 'count']))
adv20 = adv20[adv20['count'] >= 10]
print('基准日 %s | 有效股票 %d' % (d0.date(), len(day)), flush=True)

# 复现选股
fp = pd.read_parquet(f'{ROOT}/data/big/fin_panel.parquet')
fp['date'] = pd.to_datetime(fp['date'])
fp = fp[fp.date == d0].drop_duplicates('symbol').set_index('symbol')
QUAL = ['roe_ttm', 'gpa', 'ocf_ni', 'accrual', 'debt_ratio', 'roe_stab']
QDIR = np.array([1, 1, 1, -1, -1, -1])
VAL = ['bp', 'ep', 'sp', 'ocf_yield', 'fcf_yield']

u = pd.DataFrame(index=fp.index)
for i, c in enumerate(QUAL):
    v = fp[c].astype(float)
    u[c] = (v - v.mean()) / (v.std() + 1e-12) * QDIR[i]
for c in VAL:
    v = fp[c].astype(float)
    u[c] = (v - v.mean()) / (v.std() + 1e-12)
u['Q'] = u[QUAL].mean(axis=1, skipna=True)
u['V'] = u[VAL].mean(axis=1, skipna=True)
u = u.dropna(subset=['Q', 'V'])
pool = u[u.Q >= u.Q.quantile(0.60)]
pick = pool.nlargest(TOP, 'V')
print('池子 %d 只 → 选出 %d 只' % (len(pool), len(pick)), flush=True)

# 合并行情
g = pick.join(day[['mcap']], how='left').join(adv20[['mean']].rename(columns={'mean': 'adv'}), how='left')
g = g.dropna(subset=['mcap', 'adv'])
print('有行情的持仓 %d 只' % len(g), flush=True)


def pct(s, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    return [round(float(s.quantile(q)) / 1e8, 2) for q in qs]


print('\n=== ① 市值与流动性（亿元）===')
print('%-16s %8s %8s %8s %8s %8s' % ('', 'p5', 'p25', '中位', 'p75', 'p95'))
print('%-16s %s' % ('策略持仓 流通市值', pct(g.mcap)))
allm = mk[mk.date == d0]['mcap'].dropna()
print('%-16s %s' % ('全市场   流通市值', pct(allm)))
print('%-16s %s' % ('策略持仓 20日均额', pct(g.adv)))
alla = adv20['mean'].dropna()
print('%-16s %s' % ('全市场   20日均额', pct(alla)))
print('持仓市值中位数 / 全市场中位数 = %.1f 倍' % (g.mcap.median() / allm.median()))

print('\n=== ② 市值分层 ===')
bins = [0, 30, 50, 100, 300, 1000, 1e9]
labels = ['<30亿', '30-50', '50-100', '100-300', '300-1000', '>1000亿']
h = pd.cut(g.mcap / 1e8, bins=bins, labels=labels).value_counts(normalize=True).reindex(labels).fillna(0)
a = pd.cut(allm / 1e8, bins=bins, labels=labels).value_counts(normalize=True).reindex(labels).fillna(0)
for l in labels:
    print('  %-10s 策略 %5.1f%%  全市场 %5.1f%%  超配 %+5.1fpp' % (l, h[l] * 100, a[l] * 100, (h[l] - a[l]) * 100))

print('\n=== ③ 行业暴露（前 10 大 + 偏离）===')
ind_pick = pd.Series([names.get(int(IND[t0, syms.index(s)]), '-') if s in syms else '-' for s in g.index])
ind_all = pd.Series([names.get(int(IND[t0, j]), '-') for j in range(N) if M[t0, j]])
hp = ind_pick.value_counts(normalize=True)
ha = ind_all.value_counts(normalize=True)
cmp = pd.DataFrame({'策略': hp, '全市场': ha}).fillna(0)
cmp['偏离pp'] = (cmp['策略'] - cmp['全市场']) * 100
cmp = cmp.sort_values('策略', ascending=False)
print(cmp.head(10).assign(策略=lambda d: (d['策略'] * 100).round(1),
                          全市场=lambda d: (d['全市场'] * 100).round(1)).to_string())
print('行业数 %d | 最大单行业 %.1f%% | HHI %.3f（全市场 %.3f）' %
      (len(hp), hp.max() * 100, (hp ** 2).sum(), (ha ** 2).sum()))

print('\n=== ④ 容量估算（每季度换仓、单票不超过 ADV 的 X%）===')
# 等权 300 只，单票权重 1/300；单票可承载 = ADV * X% / 权重
for x in [0.05, 0.10, 0.20]:
    cap_per = g.adv * x * TOP            # 组合总资产上限（按每只票算）
    print('  单票≤ADV %2.0f%%：组合容量 %6.0f 亿元（受最小 10%% 持仓限制，中位 %6.0f 亿）' %
          (x * 100, cap_per.quantile(0.10) / 1e8, cap_per.median() / 1e8))

print('\n=== ⑤ 冲击成本（平方根模型，单边）===')
# cost_bp = 0.1 * 日波动(%) * sqrt(交易额/ADV) * 1e4
vol = 0.02   # 假设日波动 2%
for A in [1, 5, 10, 30, 100]:
    q = A * 1e8 / TOP                     # 单票交易额
    ratio = q / g.adv
    c = 0.1 * vol * np.sqrt(ratio) * 1e4
    print('  资产 %4d 亿：单票 %6.1f 万 / ADV %6.2f%% → 单边 %.1f bp，年化拖累 %.2f%%' %
          (A, q / 1e4, ratio.median() * 100, c.median(), c.median() / 1e4 * 2.6 * 100))
