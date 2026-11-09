# -*- coding: utf-8 -*-
"""SVM 残差叠加检验：线性模型 OLS + λ×(SVM 的非线性残差)
动机：svm_daily 发现 SVM 单独劣于 OLS，但其预测里 45% 无法被 8 因子线性解释，
      且该残差 OOS IC +0.018(t 6.9)。问：把它当“增量信号”叠到线性模型上能否加分？
纪律：λ 只在验证集 2023 选；2024+ 只评估；回测口径与 svm_daily 完全一致。
"""
import numpy as np, pandas as pd
from scipy import stats as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'D:/quant'
LIB = ['INTRADAY20', 'MOM120', 'MOM60', 'QUAD_ASYM', 'REV5', 'SIZE20', 'VOL20', 'OVERNIGHT20']
COST = 2e-3

pr = pd.read_parquet(f'{ROOT}/data/svm_preds.parquet')
for f in LIB:                                   # 残差回归需要原始因子值
    x = pd.read_parquet(f'{ROOT}/data/factors/{f}.parquet')[['date', 'symbol', 'value']]
    x = x.rename(columns={'value': f})
    pr = pr.merge(x, on=['date', 'symbol'], how='left')
lab = pd.read_parquet(f'{ROOT}/data/panel.parquet')[['date', 'symbol', 'label']]
df = pr.merge(lab, on=['date', 'symbol'], how='left').sort_values(['date', 'symbol']).reset_index(drop=True)
M_TR = df['date'] < '2023-01-01'
M_VA = (df['date'] >= '2023-01-01') & (df['date'] < '2024-01-01')
M_TE = df['date'] >= '2024-01-01'

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

# 逐日截面：把 SVM 预测对 8 因子回归 → 残差（= 线性无法解释的部分）
# 同时把每个分数标准化到同一尺度，便于加权叠加
def cs_z(s):
    return (s - s.mean()) / (s.std() + 1e-12)

rows = []
for dt, g in df.groupby('date'):
    g = g.dropna(subset=LIB)
    if len(g) < 30: continue
    A = np.column_stack([np.ones(len(g))] + [g[f].values for f in LIB])
    y = g['p_svm'].values
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    rows.append(pd.DataFrame({
        'idx': g.index, 'res': y - A @ b,
        'z_ols': cs_z(g['p_ols'].values), 'z_svm': cs_z(y),
        'z_icir': cs_z(g['p_icir'].values)}))
R = pd.concat(rows).set_index('idx')
df = df.join(R)

print('=== 分量诊断（OOS） ===')
for col, nm in [('z_ols', 'OLS'), ('z_svm', 'SVM'), ('res', 'SVM非线性残差'), ('z_icir', 'ICIR合成')]:
    ic = daily_ic(df[M_TE], col)
    print('  %-16s OOS IC %+.4f (ICIR %+.2f, NW-t %+.1f)' % (nm, ic.mean(), ic.mean()/ic.std(), nw_t(ic)))

print('\n=== λ 选择（验证集 2023；λ=0 即纯 OLS）===')
best = None
for lam in [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
    df['mix'] = df['z_ols'] + lam * df['res']
    ic_v = daily_ic(df[M_VA], 'mix').mean()
    ic_o = daily_ic(df[M_TE], 'mix').mean()
    flag = ''
    if best is None or ic_v > best[1]: best = (lam, ic_v); flag = ' ←'
    print('  λ=%-4.1f VAL IC %+.4f | (OOS 参考 %+.4f)%s' % (lam, ic_v, ic_o, flag))
LAM = best[0]
print('  → 选 λ=%.1f' % LAM)

df['mix'] = df['z_ols'] + LAM * df['res']

def backtest(col, top=0.2, buffer=0.4, cost=COST, start=None):
    d = df if start is None else df[df['date'] >= start]
    reb = np.sort(d['date'].unique())[::5]
    prev, recs, turns = set(), [], []
    for i, dt in enumerate(reb[:-1]):
        g = d[d['date'] == dt].set_index('symbol')
        s = g[col].dropna()
        if len(s) < 50: continue
        n = max(int(len(s) * top), 5); rk = s.rank(pct=True)
        if prev:
            keep = set(rk[rk >= 1 - buffer].index) & prev
            add = rk[(rk >= 1 - top) & (~rk.index.isin(keep))].nlargest(max(n - len(keep), 1)).index
            hold = list(keep | set(add))[:n]
        else:
            hold = list(rk.nlargest(n).index)
        nxt = d[d['date'] == reb[i + 1]]
        gross = nxt[nxt['symbol'].isin(hold)]['label'].mean()
        t = len(set(hold) - prev) / max(len(hold), 1) if prev else 1.0
        recs.append((reb[i + 1], gross, gross - t * cost * 2)); turns.append(t); prev = set(hold)
    return pd.DataFrame(recs, columns=['date', 'gross', 'net']).set_index('date'), np.mean(turns)

def stats(r, tag):
    r = r.dropna(); ann = (1 + r).prod() ** (50.4 / len(r)) - 1
    vol = r.std() * np.sqrt(50.4); nav = (1 + r).cumprod(); dd = (nav / nav.cummax() - 1).min()
    print('  %-18s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%%' % (
        tag, ann*100, vol*100, (ann-.02)/vol, dd*100))
    return dict(ann=ann, sharpe=(ann-.02)/vol, dd=dd, nav=nav)

print('\n=== 回测（5日非重叠，top20%%，buffer40%%，20bp/边）===')
res, rr = {}, {}
for col, nm in [('z_ols', 'OLS纯线性'), ('mix', 'OLS+λ残差'), ('z_svm', 'SVM单独'), ('z_icir', 'ICIR合成')]:
    r, to = backtest(col); rr[nm] = r
    print(' [全样本] %s 换手%.0f%%' % (nm, to*100))
    res[nm] = stats(r['net'], nm)
print('\n [OOS 2024+]')
for col, nm in [('z_ols', 'OLS纯线性'), ('mix', 'OLS+λ残差'), ('z_svm', 'SVM单独'), ('z_icir', 'ICIR合成')]:
    r, to = backtest(col, start='2024-01-01')
    print(' %s:' % nm, end='')
    stats(r['net'], '')

# 分年度 OOS 对比
print('\n=== 分年度费后收益（%）===')
tab = {}
for nm in ['OLS纯线性', 'OLS+λ残差', 'SVM单独', 'ICIR合成']:
    r = rr[nm]['net']; tab[nm] = ((1 + r).groupby(r.index.year).prod() - 1) * 100
print(pd.DataFrame(tab).round(1).to_string())

# 图
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
for nm, c, lw in [('OLS+λ残差', '#c0392b', 2.0), ('OLS纯线性', '#2980b9', 1.4),
                  ('SVM单独', '#e67e22', 1.2), ('ICIR合成', '#7f8c8d', 1.2)]:
    nav = res[nm]['nav']
    ax[0].plot(nav.index, nav.values, lw=lw, color=c, label='%s 夏普%.2f' % (nm, res[nm]['sharpe']))
ax[0].set_yscale('log'); ax[0].legend(frameon=False, fontsize=8.5); ax[0].grid(alpha=.25, which='both')
ax[0].set_title('SVM 非线性残差叠加 vs 纯线性（费后）', fontsize=10.5)
lams = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
va_curve, oos_curve = [], []
for lam in lams:
    df['t'] = df['z_ols'] + lam * df['res']
    va_curve.append(daily_ic(df[M_VA], 't').mean()); oos_curve.append(daily_ic(df[M_TE], 't').mean())
ax[1].plot(lams, va_curve, 'o-', color='#2980b9', label='验证集 2023')
ax[1].plot(lams, oos_curve, 's--', color='#c0392b', label='样本外 2024+')
ax[1].axvline(LAM, color='#7f8c8d', lw=.9, ls=':')
ax[1].set_xlabel('λ（残差权重）'); ax[1].set_ylabel('Rank IC'); ax[1].legend(frameon=False, fontsize=9)
ax[1].set_title('残差权重的边际贡献：λ>0 即线性模型漏掉的信息', fontsize=10.5); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig(f'{ROOT}/reports/svm_stack.png', dpi=135, bbox_inches='tight')
print('\n图已保存 reports/svm_stack.png')
