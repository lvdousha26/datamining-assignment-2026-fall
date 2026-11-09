# -*- coding: utf-8 -*-
"""Pipeline v2：扩池2096只 + 历史市值代理 + 7因子 + FM/FDR + buffer组合
数据仍有的缺陷：仅现存股票（幸存者偏差）、不复权（跳跃修补）、无行业分类、EP/BP含前视偏差
"""
import pickle, math
import numpy as np, pandas as pd

panel = pickle.load(open('a_panel3.pkl', 'rb'))
close = {c: pd.Series({pd.Timestamp(x[0]): x[1] for x in d['k']}).sort_index() for c, d in panel.items()}
C0 = pd.DataFrame(close).sort_index()
V = pd.DataFrame({c: pd.Series({pd.Timestamp(x[0]): x[2] for x in d['k']}).sort_index()
                  for c, d in panel.items()}).reindex(C0.index)

def fix(px, code):
    lim = -0.21 if code.startswith(('30', '68')) else -0.11
    r = px.pct_change(); bad = r[r < lim].index
    if not len(bad): return px
    s = np.array(px.values, float, copy=True); idx = px.index
    for d in bad: s[idx.get_loc(d):] *= 1 / (1 + r.loc[d])
    return pd.Series(s, index=idx)

C = pd.DataFrame({c: fix(C0[c].dropna(), c) for c in C0.columns}).reindex(C0.index)
# 历史市值代理：流通股数固定 = 今日流通市值/今日价
SH = {c: (panel[c]['nmc'] / panel[c]['price0']) if panel[c]['price0'] > 0 else np.nan for c in panel}
EPS = {c: (panel[c]['price0'] / panel[c]['pe']) if panel[c]['pe'] > 0 else np.nan for c in panel}
BPS = {c: (panel[c]['price0'] / panel[c]['pb']) if panel[c]['pb'] > 0 else np.nan for c in panel}

me = C.groupby([C.index.year, C.index.month]).tail(1).index
print('面板 %s ~ %s, %d 日, %d 只' % (C.index[0].date(), C.index[-1].date(), len(C), C.shape[1]))

def zs(x):
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 150: return None
    x = x.clip(x.quantile(.01), x.quantile(.99))
    return (x - x.mean()) / x.std()

def neutralize(fac, size):
    d = pd.concat([fac.rename('f'), size.rename('s')], axis=1).dropna()
    if len(d) < 150: return None
    A = np.column_stack([np.ones(len(d)), d['s'].values])
    b, *_ = np.linalg.lstsq(A, d['f'].values, rcond=None)
    return pd.Series(d['f'].values - A @ b, index=d.index)

rows = []
for j, t in enumerate(me):
    i = C.index.get_loc(t)
    if i < 260 or j == len(me) - 1: continue
    w = C.iloc[:i + 1]
    ok = (w.iloc[-250:].notna().sum() >= 200) & (w.iloc[-1] > 2) & (V.iloc[i].reindex(w.columns) > 0)
    sub = w.loc[:, ok]
    if sub.shape[1] < 300: continue
    r = sub.pct_change()
    mkt = r.mean(axis=1)
    size = np.log(pd.Series(SH).reindex(sub.columns) * sub.iloc[-1]).replace([np.inf, -np.inf], np.nan)
    f = pd.DataFrame(index=sub.columns)
    f['REV20'] = -(sub.iloc[-1] / sub.iloc[-21] - 1)
    f['VOL60'] = -r.iloc[-60:].std()
    f['TURN'] = -(V[sub.columns].iloc[-20:].mean() / V[sub.columns].iloc[-250:].mean())
    f['MOM120'] = sub.iloc[-6] / sub.iloc[-126] - 1
    beta = r.iloc[-120:].apply(lambda x: np.polyfit(mkt.iloc[-120:], x, 1)[0])
    f['IDIO'] = -((r.iloc[-120:] - np.outer(mkt.iloc[-120:], beta)).std())
    f['EP'] = pd.Series(EPS).reindex(sub.columns) / sub.iloc[-1]      # 前视：今日盈利/历史价
    f['BP'] = pd.Series(BPS).reindex(sub.columns) / sub.iloc[-1]
    f = f.replace([np.inf, -np.inf], np.nan)
    facs = {}
    for k in ['REV20', 'VOL60', 'TURN', 'MOM120', 'IDIO', 'EP', 'BP']:
        nr = neutralize(f[k], size)
        if nr is None: continue
        z = zs(nr)
        if z is not None: facs[k] = z
    if len(facs) < 7: continue
    Z = pd.DataFrame(facs).replace([np.inf, -np.inf], np.nan).dropna()
    if len(Z) < 300: continue
    fy = (C.loc[me[j + 1], Z.index] / C.loc[t, Z.index] - 1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(fy) < 300: continue
    rows.append((t, Z.loc[fy.index], fy, beta.reindex(fy.index), size.reindex(fy.index)))
print('截面数 %d, 平均股票数 %.0f' % (len(rows), np.mean([len(r[1]) for r in rows])))

names = list(rows[0][1].columns)
X = pd.concat([r[1] for r in rows])
print('因子面板 %d 行；协方差条件数 %.1f' % (len(X), np.linalg.cond(np.cov(X.values.T))))

def fm(cols):
    G = []
    for t, Z, y, beta, size in rows:
        A = np.column_stack([np.ones(len(Z)), Z[cols].values])
        m = np.isfinite(y.values)
        b, *_ = np.linalg.lstsq(A[m], y.values[m], rcond=None)
        G.append(b[1:])
    G = np.array(G); out = []
    for k, c in enumerate(cols):
        g = G[:, k]; mu = g.mean(); e = g - mu; s2 = e @ e / len(g)
        for L in range(1, 4): s2 += 2 * (1 - L / 4) * (e[L:] @ e[:-L]) / len(g)
        out.append((c, mu * 100, mu / np.sqrt(s2 / len(g))))
    return out

print('\n[Fama-MacBeth 全因子 + NW]')
res = fm(names)
for c, v, t in res: print('  %-7s 月均 %+7.3f%%  t=%5.2f' % (c, v, t))
p = np.array([2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) for _, _, t in res])
order = np.argsort(p); passed = []
for rank, k in enumerate(order, 1):
    if p[k] <= rank / len(p) * 0.05: passed.append(res[k][0])
print('  BH-FDR(5%%) 显著: %s' % (passed if passed else '无'))
print('  原始p: ' + ', '.join('%s=%.3f' % (res[k][0], p[k]) for k in range(len(p))))

# ---------- 组合：前20% + buffer ----------
print('\n[组合回测]')
prevs = None; recs = []
for t, Z, y, beta, size in rows:
    comp = Z.mean(axis=1)
    n = len(comp); k = int(n * 0.2)
    rank = comp.rank(ascending=False)
    if prevs is None:
        sel = comp.nlargest(k).index
    else:                                   # buffer：老持仓只要还在前40%就留
        keep = [c for c in prevs if c in rank.index and rank[c] <= n * 0.4]
        add = comp.drop(keep, errors='ignore').nlargest(max(0, k - len(keep))).index
        sel = pd.Index(list(keep) + list(add))
    w = pd.Series(0.0, index=Z.index)
    v = beta.reindex(sel).abs().clip(lower=0.3)
    ww = (1 / v) / (1 / v).sum()
    w.loc[sel] = ww.clip(upper=0.03)
    w = w / w.sum()
    recs.append((me[me.get_loc(t) + 1], (w * y).sum(), w))
    prevs = sel
R = pd.DataFrame([(d, r) for d, r, w in recs], columns=['d', 'r']).set_index('d')
W = [w for _, _, w in recs]
turn = pd.Series([np.abs(W[i] - W[i - 1]).sum() if i else 0 for i in range(len(W))], index=R.index)
net = R['r'] - turn * 1e-3
ew = pd.Series({me[me.get_loc(t) + 1]: y.mean() for t, Z, y, beta, size in rows})

def perf(r, label):
    r = r.dropna(); nav = (1 + r).cumprod(); yrs = len(r) / 12
    ann = nav.iloc[-1] ** (1 / yrs) - 1; vol = r.std() * 12 ** .5
    print('  %-24s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%%' % (
        label, ann * 100, vol * 100, (ann - .02) / vol, (nav / nav.cummax() - 1).min() * 100))
perf(ew, '等权基准')
perf(R['r'], '多头20%(费前)')
perf(net, '多头20%(费后)')
te = net - ew
print('  超额 %.2f%%/年, IR %.2f, 月换手 %.0f%%, 年成本 %.2f%%' % (
    (net.mean() - ew.mean()) * 12 * 100, te.mean() * 12 / (te.std() * 12 ** .5),
    turn.mean() * 100, (turn * 1e-3).mean() * 12 * 100))
