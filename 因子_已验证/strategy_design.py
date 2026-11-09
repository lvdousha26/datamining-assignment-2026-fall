# -*- coding: utf-8 -*-
"""多因子策略原型：因子正交化 -> FM增量检验 -> 协方差组合优化 -> 择时叠加 -> 多重检验
数据限制：660只(市值前700)、不复权已修补、无财报/行业 → 只能用价量因子
输出仅用于验证流程，不可作为实盘依据
"""
import pickle, math, numpy as np, pandas as pd

panel = pickle.load(open('a_panel.pkl', 'rb'))
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
me = C.groupby([C.index.year, C.index.month]).tail(1).index
SIZE = {c: np.log(panel[c]['mcap']) if panel[c]['mcap'] > 0 else np.nan for c in panel}

def zs(x):
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 100: return None
    x = x.clip(x.quantile(.01), x.quantile(.99))
    return (x - x.mean()) / x.std()

def neutralize(fac, size):
    """截面回归取残差：因子 ~ 1 + log市值"""
    d = pd.concat([fac.rename('f'), size.rename('s')], axis=1).dropna()
    if len(d) < 100: return None
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
    if sub.shape[1] < 150: continue
    r = sub.pct_change()
    mkt = r.mean(axis=1)
    f = pd.DataFrame(index=sub.columns)
    f['REV20'] = -(sub.iloc[-1] / sub.iloc[-21] - 1)          # 反转（先验正）
    f['VOL60'] = -r.iloc[-60:].std()                          # 低波（先验正）
    f['TURN'] = -(V[sub.columns].iloc[-20:].mean() / V[sub.columns].iloc[-250:].mean())  # 低换手（先验正）
    f['MOM120'] = sub.iloc[-6] / sub.iloc[-126] - 1           # 中期动量（先验正）
    beta = r.iloc[-120:].apply(lambda x: np.polyfit(mkt.iloc[-120:], x, 1)[0])
    resid = r.iloc[-120:] - np.outer(mkt.iloc[-120:], beta)
    f['IDIO'] = -resid.std()                                  # 低特质波动（先验正）
    f = f.replace([np.inf, -np.inf], np.nan)
    sz = pd.Series(SIZE).reindex(sub.columns)
    facs = {}
    for k in ['REV20', 'VOL60', 'TURN', 'MOM120', 'IDIO']:
        nr = neutralize(f[k], sz)
        if nr is None: continue
        z = zs(nr)
        if z is None: continue
        facs[k] = z
    if len(facs) < 5: continue
    Z = pd.DataFrame(facs).replace([np.inf, -np.inf], np.nan).dropna()
    if len(Z) < 150: continue
    f['fwd'] = C.loc[me[j + 1], Z.index] / C.loc[t, Z.index] - 1
    fy = f.loc[Z.index, 'fwd'].replace([np.inf, -np.inf], np.nan).dropna()
    if len(fy) < 150: continue
    rows.append((t, Z.loc[fy.index], fy, beta.reindex(fy.index), sz.reindex(fy.index)))

print('截面数 %d, 平均股票数 %.0f, %s ~ %s' % (
    len(rows), np.mean([len(x[1]) for x in rows]), rows[0][0].date(), rows[-1][0].date()))

# ---------- 1. 协方差矩阵（因子层面）----------
X = pd.concat([x[1] for x in rows])
X = X.replace([np.inf, -np.inf], np.nan).dropna()
print('因子面板有效行 %d' % len(X))
X = X.values
Xc = X - X.mean(0); S = Xc.T @ Xc / (len(X) - 1)
names = list(rows[0][1].columns)
print('\n[1] 因子协方差矩阵 条件数 %.2f（>30 说明共线严重）' % np.linalg.cond(S))
print(pd.DataFrame(S, index=names, columns=names).round(3).to_string())

# ---------- 2. Fama-MacBeth + Newey-West ----------
def fm(cols):
    G = []
    for t, Z, y, beta, sz in rows:
        A = np.column_stack([np.ones(len(Z)), Z[cols].values])
        m = np.isfinite(y.values)
        b, *_ = np.linalg.lstsq(A[m], y.values[m], rcond=None)
        G.append(b[1:])
    G = np.array(G); out = []
    for k, c in enumerate(cols):
        g = G[:, k]; mu = g.mean(); e = g - mu
        s2 = e @ e / len(g)
        for L in range(1, 4):
            s2 += 2 * (1 - L / 4) * (e[L:] @ e[:-L]) / len(g)
        out.append((c, mu * 100, mu / np.sqrt(s2 / len(g))))
    return out

print('\n[2] Fama-MacBeth（先验方向，全部5因子，NW t）:')
tstats = []
for c, v, t in fm(names):
    print('  %-7s 月均 %+6.3f%%  t=%5.2f' % (c, v, t)); tstats.append((c, t))

# ---------- 3. 多重检验校正（BH-FDR）----------
print('\n[3] BH-FDR 校正（5个检验）:')
p = np.array([2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) for _, t in tstats])
order = np.argsort(p); m = len(p)
passed = []
for rank, idx in enumerate(order, 1):
    if p[idx] <= rank / m * 0.05:
        passed.append(tstats[idx][0])
print('  原始 p: ' + ', '.join('%s=%.3f' % (tstats[i][0], p[i]) for i in range(m)))
print('  FDR 5%% 后仍显著: %s' % (passed if passed else '无'))

# ---------- 4. 组合构建：协方差 + 权重约束 ----------
def build_port(Z, beta, sz, top=60, cap=0.03):
    comp = Z.mean(axis=1)                      # 等权合成（先验方向已统一）
    sel = comp.nlargest(top).index
    vol = Z.index.isin(sel)
    w = pd.Series(0.0, index=Z.index)
    v = beta.reindex(sel).abs().clip(lower=0.3)
    ww = (1 / v) / (1 / v).sum()               # 逆beta加权（粗略风险平价）
    w.loc[sel] = ww.clip(upper=cap)
    w = w / w.sum()
    return w

# ---------- 5. 回测（含成本）+ 择时叠加 ----------
don = pd.read_pickle('sh_index.pkl')['close']
def donch_state(px, ni=90, no=40):
    hi, lo = px.rolling(ni).max().shift(1), px.rolling(no).min().shift(1)
    st = pd.Series(np.nan, index=px.index); st[px > hi] = 1.0; st[px < lo] = 0.0
    return st.ffill().fillna(0)

res = []
for idx, (t, Z, y, beta, sz) in enumerate(rows):
    w = build_port(Z, beta, sz)
    r = (w * y).sum()
    res.append((me[idx + 1], r, w))
R = pd.DataFrame(res, columns=['date', 'r', 'w']).set_index('date')
ew = pd.Series({t: y.mean() for t, Z, y, beta, sz in rows}, index=[me[i + 1] for i in range(len(rows))])

turn = pd.Series([np.abs(R['w'].iloc[i] - R['w'].iloc[i - 1]).sum() if i else 0 for i in range(len(R))], index=R.index)
cost = turn * 1e-3          # 单边 10bp（含冲击）
net = R['r'] - cost

# 择时：上证丹尼斯月度状态
st = donch_state(don).reindex(me).ffill().shift(1).reindex(R.index).fillna(0)

def perf(r, label, rf=.02):
    nav = (1 + r).cumprod(); yrs = len(r) / 12
    ann = nav.iloc[-1] ** (1 / yrs) - 1; vol = r.std() * 12 ** .5
    mdd = (nav / nav.cummax() - 1).min()
    print('  %-22s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%%' % (
        label, ann * 100, vol * 100, (ann - rf) / vol, mdd * 100))
    return ann, vol

print('\n[4] 回测（月频，单边10bp，89个月）:')
perf(ew, '基准：全市场等权')
perf(R['r'], '多因子多头60只（费前）')
perf(net, '多因子多头60只（费后）')
perf(net * st + (1 - st) * 0.02 / 12, '费后 + 上证丹尼斯择时')
print('\n平均月换手 %.0f%%，年化成本 %.2f%%，超额(费后 vs 等权) %.2f%%/年' % (
    turn.mean() * 100, cost.mean() * 12 * 100, (net.mean() - ew.mean()) * 12 * 100))
te = net - ew
print('IR = %.2f（超额/跟踪误差）' % (te.mean() * 12 / (te.std() * 12 ** .5)))
