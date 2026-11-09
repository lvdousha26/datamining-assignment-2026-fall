"""按 2025 年卖方研报复现：隔夜-日内分离系列因子
参考：东吴"日与夜的殊途同归"、东吴 RPV 四象限、中信建投隔夜-日内异象
数据：OHLCV 日线（2096只大中盘，2014-2026）
"""
import pickle
import numpy as np, pandas as pd

O, H, L, C, V = pickle.load(open('hlmv.pkl', 'rb'))
ret = C.pct_change()
gap = O / C.shift(1) - 1            # 隔夜 C2O
intra = C / O - 1                   # 日内 O2C
vr = V / V.rolling(20).mean()
vwap = (H + L + C) / 3              # VWAP 近似

def rcorr(a, b, n):
    ca = a.sub(a.rolling(n).mean()); cb = b.sub(b.rolling(n).mean())
    return ca.mul(cb).rolling(n).mean() / (a.rolling(n).std() * b.rolling(n).std() + 1e-12)

def zs(x):
    med = x.median(); mad = (x - med).abs().median()
    return ((x - med) / (1.4826 * mad + 1e-12)).clip(-3, 3)

F = {}
# --- 1. 价量四象限（东吴 RPV 核心）：隔夜 x 日内 的四种组合 ---
gu = gap > 0; gd = gap < 0; iu = intra > 0; idn = intra < 0
F['QUAD_NET'] = ((gu & iu).astype(float).rolling(20).mean()
                 - (gd & idn).astype(float).rolling(20).mean())     # 双涨-双跌
F['QUAD_WASH'] = ((gd & iu).astype(float).rolling(20).mean())        # 低开高走（洗盘）
F['QUAD_TRAP'] = ((gu & idn).astype(float).rolling(20).mean())       # 高开低走（诱多）
F['QUAD_ASYM'] = F['QUAD_WASH'] - F['QUAD_TRAP']

# --- 2. 隔夜-日内相关性（中信建投"拔河效应"）---
F['ONID_CORR'] = rcorr(gap, intra, 20)

# --- 3. 日内/隔夜 各自的价量相关（东吴"殊途同归"）---
F['RPV_DAY'] = rcorr(intra, V.pct_change(), 20)      # 日内价量
F['RPV_NIGHT'] = rcorr(gap, V.pct_change(), 20)      # 隔夜价量
F['RPV_DIFF'] = F['RPV_DAY'] - F['RPV_NIGHT']        # 两者之差（殊途同归）

# --- 4. 隔夜贡献占比（隔夜负收益之谜）---
tot = C / C.shift(20) - 1
F['ON_SHARE'] = gap.rolling(20).sum() / (tot.abs() + 1e-6)

# --- 5. 日内相对 VWAP 的偏离（聪明钱方向）---
F['VWAP_DEV'] = ((C - vwap) / vwap).rolling(20).mean()

# --- 6. 隔夜跳空的方向一致性 ---
F['GAP_CONS'] = (np.sign(gap).rolling(20).mean())

sz = np.log((V * C).rolling(250).mean() + 1)
me = C.groupby([C.index.year, C.index.month]).tail(1).index
me = me[me >= '2019-01-01']
nxt = C.reindex(me).pct_change().shift(-1); nxt.index = me

# 基准因子（用于增量检验）
BENCH = {'REV20': C / C.shift(20) - 1, 'VOL60': ret.rolling(60).std()}

print('=== 新因子第一层：IC / ICIR / FDR ===')
rows = []
from scipy import stats
for name, f in F.items():
    ics, ls = [], []
    for i in range(len(me) - 1):
        d = me[i]
        a = f.reindex([d]).iloc[0]; b = nxt.loc[d]
        m = a.notna() & b.notna() & np.isfinite(a)
        if m.sum() < 300: continue
        x, y = a[m], b[m]
        ics.append(x.corr(y, method='spearman'))
        q = pd.qcut(x.rank(method='first'), 10, labels=False)
        g = y.groupby(q).mean()
        ls.append(g.iloc[-1] - g.iloc[0])
    ics = pd.Series(ics).dropna(); ls = pd.Series(ls).dropna()
    if len(ics) < 20: continue
    icir = ics.mean() / ics.std()
    rows.append(dict(因子=name, IC=ics.mean(), ICIR=icir, t=icir * np.sqrt(len(ics)),
                     IC正=(ics > 0).mean(), 多空=(1 + ls).prod() ** (12 / len(ls)) - 1, N=len(ics)))
R = pd.DataFrame(rows)
R['p'] = 2 * (1 - stats.norm.cdf(R['t'].abs()))
srt = R.sort_values('p').reset_index(drop=True)
m = len(srt); padj = []
for i, r in srt.iterrows():
    padj.append(min(1, r['p'] * m / (i + 1)))
srt['p_adj'] = np.minimum.accumulate(padj[::-1])[::-1]
R = R.merge(srt[['因子', 'p_adj']], on='因子')
R['absICIR'] = R['ICIR'].abs()
R = R.sort_values('absICIR', ascending=False)
print('%-12s %8s %8s %7s %7s %8s %8s' % ('因子', 'IC', 'ICIR', 't', '多空年化', 'p_adj', 'FDR'))
for _, r in R.iterrows():
    print('%-12s %+8.4f %+8.3f %+7.2f %+7.1f%% %8.3f %8s' % (
        r['因子'], r['IC'], r['ICIR'], r['t'], r['多空'] * 100, r['p_adj'],
        'YES' if r['p_adj'] < 0.05 else ''))

print('\n=== 第二层：size 中性化 + 剔除 REV20/VOL60 的增量 ===')
print('%-12s %8s %8s %7s %8s %8s %7s' % ('因子', 'IC原', 'IC中性', '保留', 'IC增量', '再保留', '与REV'))
out = []
for name in R['因子']:
    f = F[name]; raw = []; neu = []; inc = []; cre = []
    for i in range(len(me) - 1):
        d = me[i]
        a = f.reindex([d]).iloc[0]; b = nxt.loc[d]
        s = sz.reindex([d]).iloc[0]
        rv = BENCH['REV20'].reindex([d]).iloc[0]; vl = BENCH['VOL60'].reindex([d]).iloc[0]
        m = a.notna() & b.notna() & s.notna() & rv.notna() & vl.notna() & np.isfinite(a)
        if m.sum() < 300: continue
        x, y = a[m], b[m]
        raw.append(x.corr(y, method='spearman'))
        cre.append(x.corr(rv[m], method='spearman'))
        A1 = np.column_stack([np.ones(m.sum()), s[m].values])
        b1, *_ = np.linalg.lstsq(A1, x.values, rcond=None)
        r1 = pd.Series(x.values - A1 @ b1, index=x.index)
        neu.append(r1.corr(y, method='spearman'))
        A2 = np.column_stack([np.ones(m.sum()), s[m].values, rv[m].values, vl[m].values])
        b2, *_ = np.linalg.lstsq(A2, x.values, rcond=None)
        r2 = pd.Series(x.values - A2 @ b2, index=x.index)
        inc.append(r2.corr(y, method='spearman'))
    o, n1, n2 = np.mean(raw), np.mean(neu), np.mean(inc)
    out.append(dict(因子=name, IC原=o, IC中性=n1, 保留1=n1 / o * 100, IC增量=n2, 保留2=n2 / o * 100,
                    与REV=np.mean(cre)))
    print('%-12s %+8.4f %+8.4f %6.0f%% %+8.4f %7.0f%% %+7.2f' % (
        name, o, n1, n1 / o * 100, n2, n2 / o * 100, np.mean(cre)))
pd.DataFrame(out).to_csv('factor_lab3_results.csv', index=False)
print('\n已保存 factor_lab3_results.csv')
