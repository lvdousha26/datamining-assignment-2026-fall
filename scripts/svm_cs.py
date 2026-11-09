# -*- coding: utf-8 -*-
"""SVM 截面选股（换目标，不是重跑回归）
与 svm_daily 的区别：
  ① 训练目标改为「截面百分位排名」和「截面分类（前30% vs 后30%）」，不再拟合收益幅度
     —— 直接针对上次发现的问题：按幅度拟合会被最强因子（小市值）绑架
  ② 评估改为多空组合（top20% 多 / bottom20% 空，等权、资金中性），剥离大盘 beta 看纯截面 alpha
纪律：训练<2023 | 验证2023 选 C/gamma | 测试2024+ 只评估；5日非重叠调仓；单边20bp
"""
import time, json
import numpy as np, pandas as pd
from scipy import stats as sps
from sklearn.svm import SVR, SVC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'D:/quant'
LIB = ['INTRADAY20', 'MOM120', 'MOM60', 'QUAD_ASYM', 'REV5', 'SIZE20', 'VOL20', 'OVERNIGHT20']
DIRN = {'INTRADAY20': -1, 'MOM120': -1, 'MOM60': -1, 'QUAD_ASYM': -1,
        'REV5': 1, 'SIZE20': 1, 'VOL20': 1, 'OVERNIGHT20': 1}
COST = 2e-3
rng = np.random.default_rng(7)

panel = pd.read_parquet(f'{ROOT}/data/panel.parquet')[['date', 'symbol', 'label']]
df = None
for f in LIB:
    x = pd.read_parquet(f'{ROOT}/data/factors/{f}.parquet')[['date', 'symbol', 'value']]
    x['value'] = x['value'] * DIRN[f]
    df = x.rename(columns={'value': f}) if df is None else df.merge(
        x.rename(columns={'value': f}), on=['date', 'symbol'], how='inner')
df = df.merge(panel, on=['date', 'symbol'], how='inner').dropna().sort_values(['date', 'symbol']).reset_index(drop=True)

def cs_prep(s):
    med = s.median(); mad = (s - med).abs().median() * 1.4826 + 1e-12
    return ((s.clip(med - 5 * mad, med + 5 * mad) - med) / mad).clip(-3, 3)
for f in LIB:
    df[f] = df.groupby('date')[f].transform(cs_prep)
df['y'] = df.groupby('date')['label'].transform(lambda s: s - s.mean())
df['rk'] = df.groupby('date')['label'].transform(lambda s: s.rank(pct=True))   # 截面百分位 0-1
df['rk_c'] = df['rk'] - 0.5                                                    # 中心化
df['r_' + 'x'] = 0

M_TR = df['date'] < '2023-01-01'
M_VA = (df['date'] >= '2023-01-01') & (df['date'] < '2024-01-01')
M_TE = df['date'] >= '2024-01-01'
print('样本 %d 行 | %d 日 | %d 只' % (len(df), df['date'].nunique(), df['symbol'].nunique()), flush=True)

# ICIR 线性基线（rank 合成），作为必须打败的门槛
for f in LIB:
    df['q_' + f] = df.groupby('date')[f].rank(pct=True)
icir = {}
for f in LIB:
    s = df[M_TR].groupby('date').apply(lambda g: sps.spearmanr(g['q_' + f], g['y']).correlation).dropna()
    icir[f] = s.mean() / s.std()
w = pd.Series({f: max(icir[f], 0) for f in LIB}); w /= w.sum()
df['p_icir'] = sum(df['q_' + f] * w[f] for f in LIB)
print('ICIR 权重:', {k: round(v, 3) for k, v in w.items() if v > 0}, flush=True)

def daily_ic(d, col):
    def f(g):
        if len(g) < 30: return np.nan
        return sps.spearmanr(g[col], g['y']).correlation
    return d.groupby('date').apply(f).dropna()

def nw_t(x, lag=4):
    x = np.asarray(x); n = len(x); e = x - x.mean(); s = e @ e / n
    for l in range(1, lag + 1):
        sw = 1 - l / (lag + 1); s += 2 * sw * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan

def show(tag, col):
    out = '  %-26s' % tag
    for nm, m in [('IS', M_TR), ('VAL', M_VA), ('OOS', M_TE)]:
        ic = daily_ic(df[m], col)
        out += ' %s IC %+.4f(%+.1f)' % (nm, ic.mean(), nw_t(ic))
    print(out, flush=True)

show('ICIR rank 合成', 'p_icir')

X = LIB
Xs_all = df[X].values

# ---------- 模型 A：SVR 拟合截面排名 ----------
print('\n=== A. SVR 目标=截面百分位排名（不再拟合收益幅度）===', flush=True)
sub = rng.choice(df[M_TR].index.values, 6000, replace=False)
grid = [(C, g) for C in (0.5, 2) for g in (0.03, 0.1)]
rowsA = []
for C, g in grid:
    t0 = time.time()
    m = SVR(C=C, gamma=g, epsilon=0.05, cache_size=800).fit(
        df.loc[sub, X].values, df.loc[sub, 'rk_c'].values)
    pv = m.predict(df.loc[M_VA, X].values)
    vd = df[M_VA].assign(p=pv)
    ic = daily_ic(vd, 'p').mean()
    rowsA.append(dict(C=C, gamma=g, val_ic=ic))
    print('  C=%-4g γ=%-5g → VAL IC %+.4f  SV %.0f%%  %.0fs' % (
        C, g, ic, m.n_support_.sum() / len(sub) * 100, time.time() - t0), flush=True)
bA = max(rowsA, key=lambda r: r['val_ic'])
print('  → C=%g γ=%g' % (bA['C'], bA['gamma']), flush=True)
fin = rng.choice(df[M_TR].index.values, 12000, replace=False)
mR = SVR(C=bA['C'], gamma=bA['gamma'], epsilon=0.05, cache_size=1000).fit(
    df.loc[fin, X].values, df.loc[fin, 'rk_c'].values)
def pred_all(model, kind='reg'):
    out = np.empty(len(df))
    for i in range(0, len(df), 60000):
        out[i:i + 60000] = model.predict(df[X].values[i:i + 60000])
        print('    预测 %d/%d' % (min(i + 60000, len(df)), len(df)), flush=True)
    return out
df['p_rank'] = pred_all(mR)
show('SVR(目标=排名)', 'p_rank')

# ---------- 模型 B：SVC 分类前30% vs 后30% ----------
print('\n=== B. SVC 二分类（截面前30%=1 / 后30%=0，中间丢弃）===', flush=True)
trm = M_TR & ((df['rk'] >= 0.7) | (df['rk'] <= 0.3))
print('  训练样本 %d 行（正类占 %.0f%%）' % (trm.sum(), (df.loc[trm, 'rk'] >= 0.7).mean() * 100), flush=True)
subB = rng.choice(df[trm].index.values, min(6000, trm.sum()), replace=False)
rowsB = []
for C, g in [(0.5, 0.03), (2, 0.03), (0.5, 0.1), (5, 0.05)]:
    t0 = time.time()
    mb = SVC(C=C, gamma=g, class_weight='balanced', cache_size=800).fit(
        df.loc[subB, X].values, (df.loc[subB, 'rk'] >= 0.7).astype(int).values)
    pv = mb.decision_function(df.loc[M_VA, X].values)
    ic = daily_ic(df[M_VA].assign(p=pv), 'p').mean()
    rowsB.append(dict(C=C, gamma=g, val_ic=ic))
    print('  C=%-4g γ=%-5g → VAL IC %+.4f  SV %.0f%%  %.0fs' % (
        C, g, ic, mb.n_support_.sum() / len(subB) * 100, time.time() - t0), flush=True)
bB = max(rowsB, key=lambda r: r['val_ic'])
print('  → C=%g γ=%g' % (bB['C'], bB['gamma']), flush=True)
mC = SVC(C=bB['C'], gamma=bB['gamma'], class_weight='balanced', cache_size=1000).fit(
    df.loc[subB, X].values, (df.loc[subB, 'rk'] >= 0.7).astype(int).values)
df['p_clf'] = pred_all(mC)
show('SVC(前30%分类)', 'p_clf')

df[['date', 'symbol', 'y', 'label', 'p_icir', 'p_rank', 'p_clf']].to_parquet(
    f'{ROOT}/data/svm_cs_preds.parquet', index=False)

# ---------- 多空组合回测 ----------
print('\n=== 多空组合（top20% 多 / bottom20% 空，等权，5日非重叠）===', flush=True)
def ls_backtest(col, start=None, q=0.2, cost=COST, longonly=False):
    d = df if start is None else df[df['date'] >= start]
    reb = np.sort(d['date'].unique())[::5]
    rec, tl, ts = [], [], []
    pl, ps = set(), set()
    for i, dt in enumerate(reb[:-1]):
        g = d[d['date'] == dt].set_index('symbol')
        s = g[col].dropna()
        if len(s) < 50: continue
        n = max(int(len(s) * q), 5); rk = s.rank(pct=True)
        L = set(rk.nlargest(n).index); S = set(rk.nsmallest(n).index)
        nx = d[d['date'] == reb[i + 1]].set_index('symbol')
        rl = nx.loc[list(L & set(nx.index)), 'label'].mean()
        rs = nx.loc[list(S & set(nx.index)), 'label'].mean()
        gl = len(L - pl) / max(len(L), 1) if pl else 1.0
        gs = len(S - ps) / max(len(S), 1) if ps else 1.0
        net_l = rl - gl * cost * 2
        net_s = -(rs) - gs * cost * 2
        rec.append((reb[i + 1], net_l, net_s, rl, rs)); tl.append(gl); ts.append(gs)
        pl, ps = L, S
    r = pd.DataFrame(rec, columns=['date', 'nl', 'ns', 'gl', 'gs']).set_index('date')
    return r, np.mean(tl), np.mean(ts)

def st(r, tag):
    r = r.dropna(); ann = (1 + r).prod() ** (50.4 / len(r)) - 1
    vol = r.std() * np.sqrt(50.4); nav = (1 + r).cumprod(); dd = (nav / nav.cummax() - 1).min()
    print('  %-22s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%% 月胜率 %2.0f%%' % (
        tag, ann * 100, vol * 100, (ann - .02) / vol, dd * 100,
        ((1 + r).groupby(r.index.to_period('M')).prod() - 1 > 0).mean() * 100), flush=True)
    return dict(ann=ann, sharpe=(ann - .02) / vol, dd=dd, nav=nav)

sizes = {}
print('\n[全样本 2019-2026]')
for col, nm in [('p_rank', 'SVR(排名目标)'), ('p_clf', 'SVC(分类)'), ('p_icir', 'ICIR合成')]:
    r, tl, ts = ls_backtest(col)
    ls = (r['nl'] + r['ns']) / 2
    print(' %s 换手 多%.0f%%/空%.0f%%' % (nm, tl * 100, ts * 100))
    sizes[nm] = st(ls, nm + ' 多空')
    sizes[nm + '_L'] = st(r['nl'], nm + ' 纯多头')

print('\n[OOS 2024+]')
for col, nm in [('p_rank', 'SVR(排名目标)'), ('p_clf', 'SVC(分类)'), ('p_icir', 'ICIR合成')]:
    r, tl, ts = ls_backtest(col, start='2024-01-01')
    ls = (r['nl'] + r['ns']) / 2
    sizes[nm + '_oos'] = st(ls, nm + ' 多空')
    sizes[nm + '_L_oos'] = st(r['nl'], nm + ' 纯多头')

# ---------- 小市值中性化 ----------
print('\n=== 小市值中性化（逐日对 SIZE20 回归取残差）===', flush=True)
sz = pd.read_parquet(f'{ROOT}/data/factors/SIZE20.parquet')[['date', 'symbol', 'value']].rename(columns={'value': 'SZ'})
df2 = df.merge(sz, on=['date', 'symbol'], how='left')
for c in ['p_rank', 'p_clf', 'p_icir']:
    out = np.full(len(df2), np.nan)
    for dt, g in df2.groupby('date'):
        if len(g) < 30: continue
        A = np.column_stack([np.ones(len(g)), g['SZ'].fillna(g['SZ'].median()).values])
        y = g[c].values
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[df2.index.get_indexer(g.index)] = y - A @ b
    df2['n_' + c] = out

def ls_bt2(d, col, start=None, q=0.2):
    d = d if start is None else d[d['date'] >= start]
    reb = np.sort(d['date'].unique())[::5]; rec = []
    for i, dt in enumerate(reb[:-1]):
        g = d[d['date'] == dt].set_index('symbol'); s = g[col].dropna()
        if len(s) < 50: continue
        n = max(int(len(s) * q), 5); rk = s.rank(pct=True)
        nx = d[d['date'] == reb[i + 1]].set_index('symbol')
        L = rk.nlargest(n).index.intersection(nx.index); S = rk.nsmallest(n).index.intersection(nx.index)
        rec.append((reb[i + 1], nx.loc[L, 'label'].mean() - nx.loc[S, 'label'].mean()))
    return pd.DataFrame(rec, columns=['d', 'r']).set_index('d')['r']

for col, nm in [('n_p_rank', 'SVR去小市值'), ('n_p_clf', 'SVC去小市值'), ('n_p_icir', 'ICIR去小市值')]:
    r = ls_bt2(df2, col, start='2024-01-01')
    ann = (1 + r).prod() ** (50.4 / len(r)) - 1; vol = r.std() * np.sqrt(50.4)
    print('  OOS %-14s 多空毛 年化 %6.2f%% 夏普 %5.2f' % (nm, ann * 100, (ann - .02) / vol), flush=True)

# ---------- 图 ----------
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
for nm, c in [('SVR(排名目标) 多空', '#c0392b'), ('SVC(分类) 多空', '#e67e22'), ('ICIR合成 多空', '#2980b9')]:
    nav = sizes[nm]['nav']
    ax[0].plot(nav.index, nav.values, lw=1.8, color=c, label='%s 夏普%.2f' % (nm.replace(' 多空', ''), sizes[nm]['sharpe']))
ax[0].set_yscale('log'); ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=.25, which='both')
ax[0].set_title('截面多空组合（去市场beta，费后）', fontsize=10.5)
names = ['SVR\n排名', 'SVC\n分类', 'ICIR\n合成']
allv = [sizes['SVR(排名目标) 多空']['sharpe'], sizes['SVC(分类) 多空']['sharpe'], sizes['ICIR合成 多空']['sharpe']]
oosv = [sizes['SVR(排名目标)_oos']['sharpe'], sizes['SVC(分类)_oos']['sharpe'], sizes['ICIR合成_oos']['sharpe']]
x = np.arange(3); ax[1].bar(x - .18, allv, .36, label='全样本', color='#95a5a6')
ax[1].bar(x + .18, oosv, .36, label='OOS 2024+', color='#c0392b')
ax[1].axhline(0, color='#2c3e50', lw=.8); ax[1].set_xticks(x); ax[1].set_xticklabels(names)
ax[1].set_ylabel('夏普'); ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=.25, axis='y')
ax[1].set_title('多空夏普对比', fontsize=10.5)
plt.tight_layout(); plt.savefig(f'{ROOT}/reports/svm_cs.png', dpi=135, bbox_inches='tight')
print('\n图已保存 reports/svm_cs.png', flush=True)
json.dump({'A': rowsA, 'B': rowsB}, open(f'{ROOT}/reports/svm_cs_grid.json', 'w'), indent=2)
