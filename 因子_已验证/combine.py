# -*- coding: utf-8 -*-
"""双层框架：横截面选股(alpha) x 时序择时(beta预算)
四组对照：纯beta / 纯择时 / 纯选股 / 双层
"""
import pickle
import numpy as np, pandas as pd

panel = pickle.load(open('a_panel.pkl', 'rb'))
close = {c: pd.Series({pd.Timestamp(x[0]): x[1] for x in d['k']}).sort_index() for c, d in panel.items()}
C0 = pd.DataFrame(close).sort_index()
V = pd.DataFrame({c: pd.Series({pd.Timestamp(x[0]): x[2] for x in d['k']}).sort_index()
                  for c, d in panel.items()}).reindex(C0.index)

def fix_jumps(px, code):
    lim = -0.21 if code.startswith(('30', '68')) else -0.11
    r = px.pct_change(); bad = r[r < lim].index
    if len(bad) == 0: return px
    s = np.array(px.values, dtype=float, copy=True); idx = px.index
    for d in bad:
        i = idx.get_loc(d); s[i:] *= 1.0 / (1.0 + r.loc[d])
    return pd.Series(s, index=idx)

C = pd.DataFrame({c: fix_jumps(C0[c].dropna(), c) for c in C0.columns}).reindex(C0.index)
me = C.groupby([C.index.year, C.index.month]).tail(1).index

def zs(x):
    x = x.dropna()
    if len(x) < 50: return None
    x = x.clip(x.quantile(.01), x.quantile(.99))
    return (x - x.mean()) / x.std()

rows = []
for j, t in enumerate(me):
    i = C.index.get_loc(t)
    if i < 260 or j == len(me) - 1: continue
    win = C.iloc[:i + 1]
    ok = (win.iloc[-250:].notna().sum() >= 200) & (win.iloc[-1] > 2) & (V.iloc[i].reindex(win.columns) > 0)
    sub = win.loc[:, ok]
    if sub.shape[1] < 150: continue
    r = sub.pct_change()
    f = pd.DataFrame(index=sub.columns)
    f['REV20'] = -(sub.iloc[-1] / sub.iloc[-21] - 1)
    f['VOL60'] = -r.iloc[-60:].std()
    f['TURN'] = -(V[sub.columns].iloc[-20:].mean() / V[sub.columns].iloc[-250:].mean())
    zs_list = [zs(f[k]) for k in ['REV20', 'VOL60', 'TURN']]
    if any(z is None for z in zs_list): continue
    zc = pd.concat(zs_list, axis=1).mean(axis=1).dropna()
    nxt = me[j + 1]
    fwd = C.loc[nxt, zc.index] / C.loc[t, zc.index] - 1
    n = len(zc)
    rows.append({
        'date': t, 'ew': fwd.mean(),
        'top': fwd[zc.nlargest(max(30, int(n * .1))).index].mean(),
        'bot': fwd[zc.nsmallest(max(30, int(n * .1))).index].mean(),
        'n': n,
    })

D = pd.DataFrame(rows).set_index('date')
print('有效月 %d, 平均股票数 %.0f, %s ~ %s' % (len(D), D['n'].mean(), D.index[0].date(), D.index[-1].date()))

# 择时信号：用等权全市场指数本身（自洽），MA125 / 丹尼斯90-40
ew_idx = (1 + D['ew']).cumprod()
def sig_ma(idx, n=12):
    return (idx > idx.rolling(n, min_periods=6).mean()).astype(float).shift(1)
def sig_don(idx, ni=9, no=4):
    hi, lo = idx.rolling(ni).max().shift(1), idx.rolling(no).min().shift(1)
    st = pd.Series(np.nan, index=idx.index); st[idx > hi] = 1.0; st[idx < lo] = 0.0
    return st.ffill().fillna(0).shift(1)

tm_ma = sig_ma(ew_idx); tm_don = sig_don(ew_idx)
tm_mix = (tm_ma + tm_don) / 2
cash = 0.02 / 12

def perf(ret, label):
    nav = (1 + ret).cumprod(); yrs = len(ret) / 12
    ann = nav.iloc[-1] ** (1 / yrs) - 1; vol = ret.std() * 12 ** .5
    mdd = (nav / nav.cummax() - 1).min()
    h = len(ret) // 2
    def sh(x):
        y = len(x) / 12; n = (1 + x).cumprod()
        return (n.iloc[-1] ** (1 / y) - 1 - .02) / (x.std() * 12 ** .5)
    print('  %-26s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%% 前/后半 %.2f/%.2f' % (
        label, ann * 100, vol * 100, (ann - .02) / vol, mdd * 100, sh(ret.iloc[:h]), sh(ret.iloc[h:])))
    return ann, vol, mdd

print('\n=== 四组对照（月频，未计交易成本）===')
perf(D['ew'], '① 纯beta 等权满仓')
perf(tm_mix * D['ew'] + (1 - tm_mix) * cash, '② 纯择时(EW+双信号)')
perf(D['top'], '③ 纯选股 前10%等权')
perf(tm_mix * D['top'] + (1 - tm_mix) * cash, '④ 双层: 选股x择时')

print('\n=== 择时信号与选股alpha的关系 ===')
alpha = D['top'] - D['ew']
print('相关性(择时仓位 vs 下月alpha) %.2f' % tm_mix.corr(alpha))
print('择时仓位分布: 满仓 %.0f%% / 半仓 %.0f%% / 空仓 %.0f%%' % (
    (tm_mix >= .9).mean() * 100, ((tm_mix > .1) & (tm_mix < .9)).mean() * 100, (tm_mix <= .1).mean() * 100))
print('选股alpha: 年化 %.2f%%  月度胜率 %.0f%%' % (alpha.mean() * 12 * 100, (alpha > 0).mean() * 100))
print('alpha 在最差10个EW月份的平均: %+.2f%%（负=选股救不了beta）' % (alpha[D['ew'].nsmallest(10).index].mean() * 100))
