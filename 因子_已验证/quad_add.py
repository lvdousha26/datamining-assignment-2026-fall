# -*- coding: utf-8 -*-
"""把 QUAD_ASYM（低开高走-高开低走，已验证可交易方向=-1）作为卫星加入策略③口径A
① 构建多头组合（因子最低10%，等权，月调，含成本）
② 与策略③ 相关性
③ 卫星权重 0/5/10/15/20% 网格：年化/波动/夏普/回撤/分年/策略③下跌月贡献
警告：股票池=今日市值前2096（幸存者偏差），多头收益系统性偏乐观；多空口径才干净
"""
import io, contextlib
import numpy as np, pandas as pd
import pickle

# ---------- 1. 策略③ 口径A（直接复用 spec3 的构建）----------
with contextlib.redirect_stdout(io.StringIO()):
    import spec3
s3 = spec3.s3A.copy()                       # 月频，PeriodIndex
LEGS = spec3.legs

# ---------- 2. QUAD 因子多头组合 ----------
O, H, L, C, V = pickle.load(open('hlmv.pkl', 'rb'))
gap = O / C.shift(1) - 1
intra = C / O - 1
gu = gap > 0; gd = gap < 0; iu = intra > 0; idn = intra < 0
WASH = (gd & iu).astype(float).rolling(20).mean()
TRAP = (gu & idn).astype(float).rolling(20).mean()
ASYM = WASH - TRAP

me_all = C.groupby([C.index.year, C.index.month]).tail(1).index
me = me_all[me_all >= '2019-01-01']
nxt = C.reindex(me).pct_change().shift(-1); nxt.index = me  # me[i]信号 -> 下月收益
nxt.index = nxt.index.to_period('M')

def long_only(f, decile=10, cost_bp=20):
    """因子最低分组等权多头，月调，按换手收成本（单边cost_bp）"""
    rets, turn, prev = {}, [], None
    for i in range(len(me) - 1):
        d = me[i]
        a = f.reindex([d]).iloc[0]; y = nxt.loc[d]
        m = a.notna() & y.notna() & np.isfinite(a)
        if m.sum() < 300: continue
        x = a[m]                            # 原始因子，多头=最低组(q==0)
        q = pd.qcut(x.rank(method='first'), decile, labels=False)
        hold = x[q == 0].index
        r = y[hold].mean()
        cur = set(hold)
        if prev is not None:
            t = len(cur - prev) / max(len(cur), 1)
            r -= t * cost_bp / 1e4 * 2      # 一进一出
            turn.append(t)
        prev = cur
        rets[d.to_period('M')] = r
    return pd.Series(rets).sort_index(), np.mean(turn)

q_ls_wash, t_w = long_only(WASH)
q_ls_asym, t_a = long_only(ASYM)
print('月均换手: WASH %.0f%%  ASYM %.0f%%（成本按单边20bp）' % (t_w * 100, t_a * 100))

def stats(r, label):
    r = r.dropna(); nav = (1 + r).cumprod(); yy = len(r) / 12
    a = nav.iloc[-1] ** (1 / yy) - 1; v = r.std() * 12 ** .5
    sh = (a - .02) / v; dd = (nav / nav.cummax() - 1).min()
    print('  %-34s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%%' % (label, a*100, v*100, sh, dd*100))
    return a, v, sh, dd

print('\n=== 单项（费后，多头10%）===')
stats(q_ls_wash, 'QUAD_WASH 多头10%')
stats(q_ls_asym, 'QUAD_ASYM 多头10%')
stats(s3, '策略③ 口径A（基准）')

idx = s3.index.intersection(q_ls_asym.index)
print('\n区间: %s ~ %s, 共 %d 月' % (idx[0], idx[-1], len(idx)))
r3, rq = s3.reindex(idx), q_ls_asym.reindex(idx)
print('月收益相关: ③ vs ASYM多头 = %.2f' % r3.corr(rq))

# 因子组合在策略③下跌月的表现
dn = r3 < 0
print('③下跌月(%d个): ③均值 %+.2f%% / ASYM多头 %+.2f%%' % (dn.sum(), r3[dn].mean()*100, rq[dn].mean()*100))
up = r3 > 0
print('③上涨月(%d个): ③均值 %+.2f%% / ASYM多头 %+.2f%%' % (up.sum(), r3[up].mean()*100, rq[up].mean()*100))

# ---------- 3. 卫星权重网格 ----------
print('\n=== 卫星权重网格（从③等比例挪出资金，费后）===')
print('%-8s %8s %7s %6s %7s %8s %8s' % ('卫星%', '年化', '波动', '夏普', '回撤', 'vs③年化', 'vs③夏普'))
a0, v0, sh0, dd0 = stats(s3, '') if False else (None,)*4
def get(r):
    r = r.dropna(); nav = (1 + r).cumprod(); yy = len(r) / 12
    return nav.iloc[-1] ** (1 / yy) - 1, r.std() * 12 ** .5, (nav / nav.cummax() - 1).min()
a3, v3, dd3 = get(s3); sh3 = (a3 - .02) / v3
print('%-8s %7.2f%% %6.2f%% %6.2f %6.1f%%' % ('0', a3*100, v3*100, sh3, dd3*100))
for w in [.05, .10, .15, .20]:
    blend = (1 - w) * s3 + w * q_ls_asym
    a, v, dd = get(blend.reindex(idx).dropna()); sh = (a - .02) / v
    print('%-8s %7.2f%% %6.2f%% %6.2f %6.1f%% %+7.2fpp %+7.2f' % (
        w*100, a*100, v*100, sh, dd*100, (a-a3)*100, sh-sh3))

# ---------- 4. 分年：③ vs 加15%卫星 ----------
w = .15
blend = ((1 - w) * s3 + w * q_ls_asym).reindex(idx).dropna()
print('\n=== 分年（15%%卫星）===')
y3 = ((1 + r3).groupby(r3.index.year).prod() - 1) * 100
yb = ((1 + blend).groupby(blend.index.year).prod() - 1) * 100
yq = ((1 + rq).groupby(rq.index.year).prod() - 1) * 100
for y in y3.index:
    print('  %d: ③ %+6.1f%%  +卫星 %+6.1f%%  (因子腿 %+6.1f%%)  Δ %+5.1fpp' % (
        y, y3[y], yb[y], yq[y], yb[y] - y3[y]))

# ---------- 5. 多空口径对照（干净口径，费前）----------
print('\n=== 对照：多空（最低-最高10%，费前，仅参考）===')
def ls_series(f):
    out = []
    for i in range(len(me) - 1):
        d = me[i]
        a = f.reindex([d]).iloc[0]; y = nxt.loc[d]
        m = a.notna() & y.notna() & np.isfinite(a)
        if m.sum() < 300: continue
        x = a[m]; y2 = y[m]
        q = pd.qcut(x.rank(method='first'), 10, labels=False)
        out.append((d.to_period('M'), y2[q == 0].mean() - y2[q == 9].mean()))
    s = pd.Series(dict(out)).sort_index()
    return s
for nm, f in [('WASH', WASH), ('ASYM', ASYM)]:
    s = ls_series(f).reindex(idx).dropna()
    a = (1 + s).prod() ** (12 / len(s)) - 1; v = s.std() * 12 ** .5
    print('  %s 多空: 年化 %.1f%% 波动 %.1f%% IR %.2f  与③相关 %.2f' % (nm, a*100, v*100, (a-.02)/v, r3.corr(s)))
