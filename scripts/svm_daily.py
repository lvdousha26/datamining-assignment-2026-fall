# -*- coding: utf-8 -*-
"""SVM 日频截面选股
纪律：
  训练 <2023 | 验证 2023（只用于选 C/gamma/epsilon） | 测试 2024-01+（只看一次）
  标签 = 5 日前向收益，截面去均值（预测超额收益）
  对照：① OLS 线性 ② Ridge ③ 8因子 ICIR 线性合成 ④ SVM 预测对因子截面回归的残差
  安慰剂：日期内打乱标签、同超参重训 → OOS IC 应≈0（检验流程本身是否造假）
  回测：5 日非重叠调仓（避免重叠标签复利虚高），top20% 等权 + 前40% buffer，单边 20bp
"""
import time, json
import numpy as np, pandas as pd
from scipy import stats as sps
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
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
rng = np.random.default_rng(42)

# ---------- 数据 ----------
panel = pd.read_parquet(f'{ROOT}/data/panel.parquet')[['date', 'symbol', 'label']]
df = None
for f in LIB:
    x = pd.read_parquet(f'{ROOT}/data/factors/{f}.parquet')[['date', 'symbol', 'value']]
    x['value'] = x['value'] * DIRN[f]           # 统一为“越大越好”
    x = x.rename(columns={'value': f})
    df = x if df is None else df.merge(x, on=['date', 'symbol'], how='inner')
df = df.merge(panel, on=['date', 'symbol'], how='inner').dropna()
df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

def cs_prep(s):
    med = s.median(); mad = (s - med).abs().median() * 1.4826 + 1e-12
    return ((s.clip(med - 5 * mad, med + 5 * mad) - med) / mad).clip(-3, 3)
for f in LIB:
    df[f] = df.groupby('date')[f].transform(cs_prep)
df['y'] = df.groupby('date')['label'].transform(lambda s: s - s.mean())
for f in LIB:                                   # 逐日 rank（0-1，越大越好），线性合成用
    df['r_' + f] = df.groupby('date')[f].rank(pct=True)

M_TR = df['date'] < '2023-01-01'
M_VA = (df['date'] >= '2023-01-01') & (df['date'] < '2024-01-01')
M_TE = df['date'] >= '2024-01-01'
tr, va, te = df[M_TR], df[M_VA], df[M_TE]       # 仅用于取训练/验证索引，评估时重新切片
print('样本 %d 行 | %d 日 | %d 只' % (len(df), df['date'].nunique(), df['symbol'].nunique()))
print('训练 %d 行(%s~%s) | 验证 %d 行(2023) | 测试 %d 行(2024+)' % (
    len(tr), tr['date'].min().date(), tr['date'].max().date(), len(va), len(te)))

YSD = tr['y'].std()      # 目标标准化尺度，SVR 的 epsilon 才有直观含义
PARTS = lambda: [('IS', df[M_TR]), ('VAL', df[M_VA]), ('OOS', df[M_TE])]

def daily_ic(d, col):
    def f(g):
        if len(g) < 30: return np.nan
        return sps.spearmanr(g[col], g['y']).correlation
    return d.groupby('date').apply(f).dropna()

def nw_t(x, lag=4):
    x = np.asarray(x); n = len(x); e = x - x.mean(); s = e @ e / n
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1); s += 2 * w * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan

def ic_stats(d, col):
    ic = daily_ic(d, col)
    return ic.mean(), (ic.mean() / ic.std() if ic.std() > 0 else np.nan), nw_t(ic), ic

def show(tag, col, extra=''):
    line = '  %-34s' % tag
    for nm, d in PARTS():
        m, ir, t, _ = ic_stats(d, col)
        line += ' %s IC %+.4f(%+.1f) ' % (nm, m, t)
    print(line + extra)

# ---------- ① 线性基线 ----------
print('\n=== ① 线性基线 ===')
X = LIB
ols = LinearRegression().fit(tr[X], tr['y'])
df['p_ols'] = ols.predict(df[X])
show('OLS', 'p_ols')

rdg = Ridge(alpha=10.0).fit(tr[X], tr['y'])
df['p_ridge'] = rdg.predict(df[X])
show('Ridge(alpha=10)', 'p_ridge')

# 8 因子 ICIR 线性合成（权重只用训练期）
icir = {}
for f in LIB:
    icir[f] = daily_ic(df[M_TR], 'r_' + f).mean() / daily_ic(df[M_TR], 'r_' + f).std()
w = pd.Series({f: max(icir[f], 0) for f in LIB}); w /= w.sum()
print('\n  训练期 ICIR 权重:', {k: round(v, 3) for k, v in w.items() if v > 0})
df['p_icir'] = sum(df['r_' + f] * w[f] for f in LIB)
show('8因子 ICIR 合成', 'p_icir')

# ---------- ② SVR 网格（训练子样本，验证集选超参） ----------
print('\n=== ② RBF-SVR 网格（训练子样本 8k，验证集 2023 选参）===', flush=True)
N_SUB = 8000
sub_idx = rng.choice(tr.index.values, size=min(N_SUB, len(tr)), replace=False)
Xs, ys = tr.loc[sub_idx, X].values, tr.loc[sub_idx, 'y'].values / YSD
Xv = va[X].values
grid = [(C, g, 0.2) for C in (0.5, 2, 8) for g in (0.05, 0.2)]
rows = []
t0 = time.time()
for C, g, e in grid:
    t1 = time.time()
    m = SVR(C=C, gamma=g, epsilon=e, cache_size=800).fit(Xs, ys)
    vd = va.assign(p=m.predict(Xv))
    ic_v = daily_ic(vd, 'p').mean()
    rows.append(dict(C=C, gamma=str(g), eps=e, val_ic=round(ic_v, 5), nSV=m.n_support_.sum() / len(Xs)))
    print('  C=%-4g gamma=%-6s eps=%-4g → VAL IC %+.4f  SV占比 %.0f%%  %.0fs' % (
        C, g, e, ic_v, m.n_support_.sum() / len(Xs) * 100, time.time() - t1), flush=True)
print('  网格耗时 %.0fs' % (time.time() - t0), flush=True)

best = max(rows, key=lambda r: r['val_ic'])
print('  → 最优 C=%g gamma=%s eps=%g（VAL IC %+.4f）' % (best['C'], best['gamma'], best['eps'], best['val_ic']), flush=True)
BG = float(best['gamma'])

# 用最优超参在更大子样本(12k)上重训
N_FIN = 12000
fin_idx = rng.choice(tr.index.values, size=min(N_FIN, len(tr)), replace=False)
t0 = time.time()
svm = SVR(C=best['C'], gamma=BG, epsilon=best['eps'], cache_size=1000).fit(
    tr.loc[fin_idx, X].values, tr.loc[fin_idx, 'y'].values / YSD)
print('  最终模型：%d 行训练，%.0fs，支持向量 %.0f%%' % (
    len(fin_idx), time.time() - t0, svm.n_support_.sum() / len(fin_idx) * 100), flush=True)

def svm_pred(frame, model, tag):
    out = np.empty(len(frame), dtype=np.float64)
    t0 = time.time()
    for i in range(0, len(frame), 60000):
        out[i:i + 60000] = model.predict(frame[X].values[i:i + 60000])
        print('    %s 预测 %d/%d (%.0fs)' % (tag, min(i + 60000, len(frame)), len(frame), time.time() - t0), flush=True)
    return out

df['p_svm'] = svm_pred(df, svm, 'SVM')
show('RBF-SVR(最优)', 'p_svm')

# ---------- ③ 全样本核岭（Nystroem 近似，验证“子样本”不是瓶颈） ----------
print('\n=== ③ 全样本核方法（Nystroem 120 维近似 + Ridge，检验子样本不是瓶颈）===', flush=True)
ny = Nystroem(gamma=BG, n_components=120, random_state=0).fit(tr[X].values)
Ztr = ny.transform(tr[X].values).astype(np.float32)
kr = Ridge(alpha=1.0).fit(Ztr, tr['y'].values)
del Ztr
def kern_pred(frame):
    out = np.empty(len(frame), dtype=np.float64)
    for i in range(0, len(frame), 100000):
        blk = frame[X].values[i:i + 100000]
        out[i:i + 100000] = kr.predict(ny.transform(blk).astype(np.float32))
    return out
df['p_krr'] = kern_pred(df)
show('核岭(Nystroem120,全样本)', 'p_krr')

# ---------- ④ 残差检验：SVM 是否只是因子的线性组合 ----------
print('\n=== ④ 残差检验（SVM 预测对 8 因子做逐日截面回归，取残差）===')
def resid(d, col):
    out = []
    for dt, g in d.groupby('date'):
        if len(g) < 30: continue
        A = np.column_stack([np.ones(len(g))] + [g[f].values for f in X])
        y = g[col].values
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ b
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        out.append(pd.DataFrame({'date': dt, 'symbol': g['symbol'].values,
                                 'y': g['y'].values, 'res': y - pred, 'r2': r2}))
    return pd.concat(out, ignore_index=True)

for tag, col in [('SVM', 'p_svm'), ('Ridge', 'p_ridge')]:
    rr = resid(df, col)
    m, ir, t, _ = ic_stats(rr, 'res')
    print('  %-6s 原始 OOS IC %+.4f → 残差 IC %+.4f（t %+.1f）  线性可解释 R² = %.2f' % (
        tag, ic_stats(df[df['date'] >= '2024-01-01'], col)[0], m, t, rr['r2'].mean()))

# ---------- ⑤ 安慰剂：日期内打乱标签 ----------
print('\n=== ⑤ 安慰剂检验（训练集日期内打乱标签，同超参重训）===', flush=True)
perm = tr.loc[fin_idx].copy()
perm['y'] = perm.groupby('date')['y'].transform(lambda s: rng.permutation(s.values))
svm_p = SVR(C=best['C'], gamma=BG, epsilon=best['eps'], cache_size=1000).fit(
    perm[X].values, perm['y'].values / YSD)
df['p_placebo'] = svm_pred(df, svm_p, '安慰剂')
show('打乱标签的 SVM（应≈0）', 'p_placebo')

# 保存预测，供后续复用
df[['date', 'symbol', 'y'] + ['p_' + x for x in ['ols', 'ridge', 'icir', 'svm', 'krr', 'placebo']]].to_parquet(
    f'{ROOT}/data/svm_preds.parquet', index=False)

# ---------- ⑥ 组合回测 ----------
print('\n=== ⑥ 组合回测（5 日非重叠，top20%%，buffer40%%，单边 20bp）===')
def backtest(col, top=0.2, buffer=0.4, cost=COST, start=None):
    d = df if start is None else df[df['date'] >= start]
    dates = np.sort(d['date'].unique())
    reb = dates[::5]
    prev, recs, turns, nh = set(), [], [], []
    for i, dt in enumerate(reb[:-1]):
        g = d[d['date'] == dt].set_index('symbol')
        if len(g) < 50: continue
        s = g[col].dropna()
        if len(s) < 50: continue
        n = max(int(len(s) * top), 5)
        rk = s.rank(pct=True)
        if prev:
            keep = set(rk[rk >= 1 - buffer].index) & prev
            need = n - len(keep)
            add = rk[(rk >= 1 - top) & (~rk.index.isin(keep))].nlargest(max(need, 1)).index
            hold = list(keep | set(add))[:n]
        else:
            hold = list(rk.nlargest(n).index)
        nxt = d[d['date'] == reb[i + 1]]
        gross = nxt[nxt['symbol'].isin(hold)]['label'].mean()
        t = len(set(hold) - prev) / max(len(hold), 1) if prev else 1.0
        recs.append((reb[i + 1], gross, gross - t * cost * 2))
        turns.append(t); nh.append(len(hold)); prev = set(hold)
    r = pd.DataFrame(recs, columns=['date', 'gross', 'net']).set_index('date')
    return r, np.mean(turns), np.mean(nh)

def stats(r, tag):
    r = r.dropna()
    ann = (1 + r).prod() ** (50.4 / len(r)) - 1
    vol = r.std() * np.sqrt(50.4)
    nav = (1 + r).cumprod(); dd = (nav / nav.cummax() - 1).min()
    print('  %-12s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%% 胜率 %2.0f%%' % (
        tag, ann * 100, vol * 100, (ann - .02) / vol, dd * 100, (r > 0).mean() * 100))
    return dict(ann=ann, vol=vol, sharpe=(ann - .02) / vol, dd=dd, nav=nav)

res = {}
for col, nm in [('p_svm', 'SVM'), ('p_krr', '核岭'), ('p_ridge', 'Ridge'), ('p_icir', 'ICIR合成')]:
    r, to, nh = backtest(col)
    print(' [全样本] %s' % nm)
    res[nm] = stats(r['net'], nm + '(费后)')
    stats(r['gross'], nm + '(费前)')
    print('             换手/期 %.0f%% 持仓 %.0f 只' % (to * 100, nh))

ew = df.groupby('date')['label'].mean()
ew = ew.loc[ew.index.isin(np.sort(df['date'].unique())[::5])]
print(' [基准] 等权全市场'); stats(ew.dropna(), '等权基准')

print('\n [OOS 2024+]')
for col, nm in [('p_svm', 'SVM'), ('p_krr', '核岭'), ('p_ridge', 'Ridge'), ('p_icir', 'ICIR合成')]:
    r, to, nh = backtest(col, start='2024-01-01')
    res[nm + '_oos'] = stats(r['net'], nm + '(费后)')
    stats(r['gross'], nm + '(费前)')

# ---------- 图 ----------
fig, ax = plt.subplots(2, 1, figsize=(11.5, 8.4), height_ratios=[3, 1.5])
for nm, c, lw in [('SVM', '#c0392b', 2.0), ('核岭', '#e67e22', 1.6),
                  ('Ridge', '#2980b9', 1.4), ('ICIR合成', '#7f8c8d', 1.2)]:
    nav = res[nm]['nav']
    ax[0].plot(nav.index, nav.values, lw=lw, color=c, label='%s 夏普%.2f' % (nm, res[nm]['sharpe']))
ewn = (1 + ew.dropna()).cumprod()
ax[0].plot(ewn.index, ewn.values, lw=1.0, color='#2c3e50', ls='--', label='等权基准')
ax[0].set_yscale('log'); ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=.25, which='both')
ax[0].set_title('SVM 日频选股 vs 线性模型（5日调仓，费后20bp）', fontsize=11)
nav = res['SVM']['nav']
ax[1].fill_between(nav.index, (nav / nav.cummax() - 1) * 100, 0, color='#c0392b', alpha=.35, lw=0)
ax[1].set_ylabel('回撤 %'); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig(f'{ROOT}/reports/svm_nav.png', dpi=135, bbox_inches='tight')
print('\n图已保存 reports/svm_nav.png')

json.dump({'best': best, 'grid': rows}, open(f'{ROOT}/reports/svm_grid.json', 'w'), ensure_ascii=False, indent=2)
