# -*- coding: utf-8 -*-
"""截面多空评估（复用 svm_cs 的预测，不重训）
把毛/净分开，因为三模型换手差 2 倍；小市值中性化单列，避免口径混淆。
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'D:/quant'
COST = 2e-3
df = pd.read_parquet(f'{ROOT}/data/svm_cs_preds.parquet')
sz = pd.read_parquet(f'{ROOT}/data/factors/SIZE20.parquet')[['date', 'symbol', 'value']].rename(columns={'value': 'SZ'})
df = df.merge(sz, on=['date', 'symbol'], how='left')

# 小市值中性化（逐日截面回归取残差）
for c in ['p_rank', 'p_clf', 'p_icir']:
    out = np.full(len(df), np.nan)
    for dt, g in df.groupby('date'):
        if len(g) < 30: continue
        x = g['SZ'].fillna(g['SZ'].median()).values; y = g[c].values
        A = np.column_stack([np.ones(len(g)), x]); b, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[df.index.get_indexer(g.index)] = y - A @ b
    df['n_' + c] = out

def ls_bt(col, start=None, q=0.2, cost=COST, neutral=False):
    d = df if start is None else df[df['date'] >= start]
    reb = np.sort(d['date'].unique())[::5]
    rec = []; pl, ps = set(), set()
    for i, dt in enumerate(reb[:-1]):
        g = d[d['date'] == dt].set_index('symbol'); s = g[col].dropna()
        if len(s) < 50: continue
        n = max(int(len(s) * q), 5); rk = s.rank(pct=True)
        L = set(rk.nlargest(n).index); S = set(rk.nsmallest(n).index)
        nx = d[d['date'] == reb[i + 1]].set_index('symbol')
        rl = nx.loc[list(L & set(nx.index)), 'label'].mean()
        rs = nx.loc[list(S & set(nx.index)), 'label'].mean()
        tl = len(L - pl) / max(len(L), 1) if pl else 1.0
        ts = len(S - ps) / max(len(S), 1) if ps else 1.0
        gross = (rl - rs) / 2
        tx = (tl + ts) / 2 * cost * 2
        rec.append((reb[i + 1], gross, gross - tx, rl - tl * cost * 2)); pl, ps = L, S
    return pd.DataFrame(rec, columns=['date', 'gross', 'net', 'nl']).set_index('date'), (tl + ts) / 2

def st(r, tag):
    r = r.dropna(); ann = (1 + r).prod() ** (50.4 / len(r)) - 1
    vol = r.std() * np.sqrt(50.4); nav = (1 + r).cumprod(); dd = (nav / nav.cummax() - 1).min()
    sh = (ann - .02) / vol
    print('  %-30s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%%' % (tag, ann*100, vol*100, sh, dd*100), flush=True)
    return dict(ann=ann, vol=vol, sharpe=sh, dd=dd, nav=nav)

MODELS = [('p_rank', 'SVR排名'), ('p_clf', 'SVC分类'), ('p_icir', 'ICIR合成')]
out = {}
for tag, s, sfx in [('全样本 2019-2026', None, ''), ('OOS 2024+', '2024-01-01', '_oos')]:
    print('\n=== %s ===' % tag)
    for col, nm in MODELS:
        r, to = ls_bt(col, start=s)
        print(' %s 换手/期 %.0f%%' % (nm, to * 100), flush=True)
        out[nm + sfx + '_g'] = st(r['gross'], nm + ' 多空(毛)')
        out[nm + sfx + '_n'] = st(r['net'], nm + ' 多空(净)')

print('\n=== 小市值中性化（多空，净）===')
for col, nm in MODELS:
    r, to = ls_bt('n_' + col, start='2024-01-01')
    out[nm + '_neu'] = st(r['net'], nm + ' 去SIZE20(净)')

# 图：全样本净多空净值 + 三种口径夏普
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
for nm, c, lw in [('ICIR合成', '#2980b9', 2.0), ('SVC分类', '#e67e22', 1.5), ('SVR排名', '#c0392b', 1.5)]:
    nav = out[nm + '_n']['nav']
    ax[0].plot(nav.index, nav.values, lw=lw, color=c, label='%s 夏普%.2f' % (nm, out[nm + '_n']['sharpe']))
ax[0].axhline(1, color='#7f8c8d', lw=.8, ls='--')
ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=.25)
ax[0].set_title('截面多空净值（费后，市场中性）', fontsize=10.5)

x = np.arange(3); names = [m[1] for m in MODELS]
v_g = [out[nm + '_g']['sharpe'] for _, nm in MODELS]
v_n = [out[nm + '_n']['sharpe'] for _, nm in MODELS]
v_o = [out[nm + '_oos_n']['sharpe'] for _, nm in MODELS]
ax[1].bar(x - .25, v_g, .25, label='毛(全样本)', color='#bdc3c7')
ax[1].bar(x, v_n, .25, label='净(全样本)', color='#2980b9')
ax[1].bar(x + .25, v_o, .25, label='净(OOS)', color='#c0392b')
ax[1].axhline(0, color='#2c3e50', lw=.8); ax[1].set_xticks(x); ax[1].set_xticklabels(names)
ax[1].set_ylabel('多空夏普'); ax[1].legend(frameon=False, fontsize=8.5); ax[1].grid(alpha=.25, axis='y')
ax[1].set_title('成本吃掉多少：毛 vs 净', fontsize=10.5)
plt.tight_layout(); plt.savefig(f'{ROOT}/reports/svm_cs.png', dpi=135, bbox_inches='tight')
print('\n图已保存 reports/svm_cs.png')

# 分年度多空净收益
print('\n=== 分年度多空净收益（%）===')
tab = {}
for col, nm in MODELS:
    r, _ = ls_bt(col)
    tab[nm] = ((1 + r['net']).groupby(r.index.year).prod() - 1) * 100
print(pd.DataFrame(tab).round(1).to_string())
