"""第二层：中性化 + 正交性 + 合成，检验哪些是真因子"""
import pickle
import numpy as np, pandas as pd
from scipy import stats

O, H, L, C, V = pickle.load(open('hlmv.pkl', 'rb'))
ret = C.pct_change()
gap = O / C.shift(1) - 1
intra = C / O - 1
mkt = ret.mean(axis=1)
vr = V / V.rolling(20).mean()

def rcorr(a, b, n):
    ca = a.sub(a.rolling(n).mean()); cb = b.sub(b.rolling(n).mean())
    return ca.mul(cb).rolling(n).mean() / (a.rolling(n).std() * b.rolling(n).std() + 1e-12)

cols20 = [c for c in C.columns if c.startswith(('30', '68'))]
fac = pd.DataFrame(1.1, index=C.index, columns=C.columns)
fac.loc[C.index >= '2020-08-24', cols20] = 1.2
islim = (C - (C.shift(1) * fac).round(2)).abs() < 0.011

F = {
 'AMIHUD20': (ret.abs() / (V * C / 1e8)).rolling(20).mean(),
 'LIMITUP60': islim.astype(float).rolling(60).sum(),
 'VOLSTAB': V.rolling(20).mean() / V.rolling(250).mean(),
 'INTRADAY20': intra.rolling(20).sum(),
 'CORRPV20': rcorr(ret, V.pct_change(), 20),
 'OVERNIGHT20': gap.rolling(20).sum(),
 'MAXRET20': ret.rolling(20).max(),
 'UPPERSH20': ((H - np.maximum(O, C)) / C).rolling(20).mean(),
 'GAPABS20': gap.abs().rolling(20).mean(),
 'MINRET20': ret.rolling(20).min(),
}
# size 代理：过去250日平均成交额（对数）
sz = np.log((V * C).rolling(250).mean() + 1)

me = C.groupby([C.index.year, C.index.month]).tail(1).index
me = me[me >= '2019-01-01']
nxt = C.reindex(me).pct_change().shift(-1); nxt.index = me

def mad_z(x):
    med = x.median(); mad = (x - med).abs().median()
    z = (x - med) / (1.4826 * mad + 1e-12)
    return z.clip(-3, 3)

res = {}
for name, f in F.items():
    fm = f.reindex(me); sm = sz.reindex(me)
    ic_raw, ic_neu, ics_pairs = [], [], {k: [] for k in F}
    for i in range(len(me) - 1):
        d = me[i]
        a, b, s = fm.loc[d], nxt.loc[d], sm.loc[d]
        m = a.notna() & b.notna() & s.notna() & np.isfinite(a) & np.isfinite(s)
        if m.sum() < 300: continue
        x, y, sv = a[m], b[m], s[m]
        ic_raw.append(x.corr(y, method='spearman'))
        # 对 size 回归取残差
        A = np.column_stack([np.ones(len(sv)), sv.values])
        beta, *_ = np.linalg.lstsq(A, x.values, rcond=None)
        resid = pd.Series(x.values - A @ beta, index=x.index)
        ic_neu.append(resid.corr(y, method='spearman'))
        # 与其他因子的截面相关
        for k2, f2 in F.items():
            v2 = f2.reindex([d]).iloc[0]
            mm = resid.index.intersection(v2.dropna().index)
            if len(mm) > 200:
                ics_pairs[k2].append(resid.loc[mm].corr(v2.loc[mm], method='spearman'))
    ic_raw, ic_neu = pd.Series(ic_raw).dropna(), pd.Series(ic_neu).dropna()
    res[name] = dict(IC_raw=ic_raw.mean(), ICIR_raw=ic_raw.mean() / ic_raw.std(),
                     t_raw=ic_raw.mean() / ic_raw.std() * np.sqrt(len(ic_raw)),
                     IC_neu=ic_neu.mean(), ICIR_neu=ic_neu.mean() / ic_neu.std(),
                     t_neu=ic_neu.mean() / ic_neu.std() * np.sqrt(len(ic_neu)),
                     保留=ic_neu.mean() / ic_raw.mean() if abs(ic_raw.mean()) > 1e-6 else np.nan,
                     corr_size=x.corr(sv, method='spearman') if False else np.nan,
                     pairs={k2: np.nanmean(v) for k2, v in ics_pairs.items()})

T = pd.DataFrame(res).T
# size 相关单独算
for name, f in F.items():
    cs = []
    for d in me[:-1]:
        a, s = f.reindex([d]).iloc[0], sz.reindex([d]).iloc[0]
        m = a.notna() & s.notna()
        if m.sum() > 300: cs.append(a[m].corr(s[m], method='spearman'))
    T.loc[name, 'corr_size'] = np.nanmean(cs)

T['|ICIR_neu|'] = T['ICIR_neu'].abs()
T = T.sort_values('|ICIR_neu|', ascending=False)
print('=== 中性化前后对比（%d 个月）===' % len(ic_neu))
print('%-11s %7s %7s %7s %7s %7s %7s' % ('因子', 'IC原', 'IC中性', '保留%', '与sizeρ', 't原', 't中性'))
for n, r in T.iterrows():
    print('%-11s %7.4f %7.4f %6.0f%% %8.2f %7.2f %7.2f' % (
        n, r['IC_raw'], r['IC_neu'], r['保留'] * 100, r['corr_size'], r['t_raw'], r['t_neu']))

print('\n=== 中性化后的因子间相关（|ρ|>0.5 视为同一因子）===')
keep = T[T['|ICIR_neu|'] > 0.15].index.tolist()
P = pd.DataFrame({n: {k: v for k, v in T.loc[n, 'pairs'].items()} for n in keep}).T
print(P[keep].round(2).to_string())

# ---------- 合成 ----------
print('\n=== 合成（按中性化后显著因子，方向按IC符号对齐）===')
sel = [n for n in keep if abs(T.loc[n, 't_neu']) > 1.5]
print('入选:', sel)
W = {n: np.sign(T.loc[n, 'IC_neu']) for n in sel}
prev = None; R_ls, R_lg = {}, {}
turns = []
for i in range(len(me) - 1):
    d = me[i]
    comp = None
    for n in sel:
        a = F[n].reindex([d]).iloc[0]; s = sz.reindex([d]).iloc[0]
        m = a.notna() & s.notna() & np.isfinite(a)
        if m.sum() < 300: continue
        A = np.column_stack([np.ones(m.sum()), s[m].values])
        beta, *_ = np.linalg.lstsq(A, a[m].values, rcond=None)
        r = pd.Series(a[m].values - A @ beta, index=a[m].index)
        z = mad_z(r) * W[n]
        comp = z if comp is None else comp.add(z, fill_value=0)
    if comp is None or len(comp) < 300: continue
    y = nxt.loc[d].reindex(comp.dropna().index).dropna()
    c2 = comp.reindex(y.index)
    q = pd.qcut(c2.rank(method='first'), 10, labels=False)
    R_ls[d] = y[q == 9].mean() - y[q == 0].mean()
    R_lg[d] = y[q == 9].mean() - y.mean()
    top = set(c2[q == 9].index)
    if prev is not None:
        turns.append(len(top - prev) / max(len(top), 1))
    prev = top

ls = pd.Series(R_ls).sort_index(); lg = pd.Series(R_lg).sort_index()
def st(x, lab):
    nav = (1 + x).cumprod(); y = len(x) / 12
    a = nav.iloc[-1] ** (1 / y) - 1; v = x.std() * 12 ** .5
    print('  %-18s 年化%6.2f%% 波动%5.2f%% 夏普%5.2f 回撤%6.1f%% 胜率%.0f%%' % (
        lab, a * 100, v * 100, (a - .02) / v, (nav / nav.cummax() - 1).min() * 100, (x > 0).mean() * 100))
st(ls, '多空(10-1组)')
st(lg, '多头超额(前10%-均值)')
print('  月均单边换手 %.0f%% → 年成本约 %.2f%%(双边20bp)' % (
    np.mean(turns) * 100, np.mean(turns) * 12 * 0.002 * 100))
ann_ls = (1 + ls).prod() ** (12 / len(ls)) - 1
net = ann_ls - np.mean(turns) * 12 * 0.002
print('  多空费后年化 %.2f%%' % (net * 100))
# 分段
h = len(ls) // 2
for seg, lab in [(ls.iloc[:h], '前段'), (ls.iloc[h:], '后段')]:
    a = (1 + seg).prod() ** (12 / len(seg)) - 1
    print('  %s(%d月): 多空年化 %.1f%%' % (lab, len(seg), a * 100))

T.drop(columns=['pairs']).to_csv('factor_lab2_results.csv')
print('\n已保存 factor_lab2_results.csv')
