# -*- coding: utf-8 -*-
"""决策树（含剪枝）做截面收益预测
设计：
  训练 < 2024-01-01，验证 2023 全年（选剪枝强度），测试 2024-01-01+
  标签做截面去均值（预测超额收益），评估用日频 Rank IC
  对照：①不剪枝的深度扫描（看 overfit 曲线）②线性 OLS ③ 8因子 rank 合成
剪枝：ccp_alpha（代价复杂度）+ max_depth + min_samples_leaf 三重约束
"""
import json
import numpy as np, pandas as pd
from scipy import stats as sps
from sklearn.tree import DecisionTreeRegressor, export_text
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'D:/quant'
LIB = ['INTRADAY20', 'MOM120', 'MOM60', 'QUAD_ASYM', 'REV5', 'SIZE20', 'VOL20', 'OVERNIGHT20']
DIRN = {'INTRADAY20': -1, 'MOM120': -1, 'MOM60': -1, 'QUAD_ASYM': -1,
        'REV5': 1, 'SIZE20': 1, 'VOL20': 1, 'OVERNIGHT20': 1}

panel = pd.read_parquet(f'{ROOT}/data/panel.parquet')[['date', 'symbol', 'label']]
df = None
for f in LIB:
    x = pd.read_parquet(f'{ROOT}/data/factors/{f}.parquet')[['date', 'symbol', 'value']]
    x['value'] = x['value'] * DIRN[f]                     # 统一方向：越大越好
    x = x.rename(columns={'value': f})
    df = x if df is None else df.merge(x, on=['date', 'symbol'], how='inner')
df = df.merge(panel, on=['date', 'symbol'], how='inner').dropna()
print('样本 %d 行, %d 日, %d 只' % (len(df), df['date'].nunique(), df['symbol'].nunique()))

# 截面预处理：5×MAD 截尾 + z-score（逐日）
def cs_prep(s):
    med = s.median(); mad = (s - med).abs().median() * 1.4826 + 1e-12
    return ((s.clip(med - 5*mad, med + 5*mad) - med) / mad).clip(-3, 3)
for f in LIB:
    df[f] = df.groupby('date')[f].transform(cs_prep)
df['y'] = df.groupby('date')['label'].transform(lambda s: s - s.mean())   # 截面超额

tr = df[df['date'] < '2023-01-01']
va = df[(df['date'] >= '2023-01-01') & (df['date'] < '2024-01-01')]
te = df[df['date'] >= '2024-01-01']
X = LIB
print('训练 %d 行 | 验证 %d 行 | 测试 %d 行' % (len(tr), len(va), len(te)))

def daily_ic(d, pred_col):
    def f(g):
        if len(g) < 30: return np.nan
        return sps.spearmanr(g[pred_col], g['y']).correlation
    ic = d.groupby('date').apply(f).dropna()
    return ic

def nw_t(x, lag=4):
    x = np.asarray(x); n = len(x); e = x - x.mean()
    s = e @ e / n
    for l in range(1, lag+1):
        w = 1 - l/(lag+1); s += 2*w*(e[l:] @ e[:-l])/n
    return x.mean()/np.sqrt(s/n) if s > 0 else np.nan

def evaluate(model, tag, verbose=True):
    out = {}
    for nm, d in [('IS', tr), ('VAL', va), ('OOS', te)]:
        p = model.predict(d[X])
        dd = d.assign(pred=p)
        ic = daily_ic(dd, 'pred')
        out[nm] = (ic.mean(), ic.mean()/ic.std() if ic.std() > 0 else np.nan, nw_t(ic))
    if verbose:
        print('  %-30s IS IC %+.4f | VAL %+.4f | OOS IC %+.4f (ICIR %+.2f, NW-t %+.1f)' % (
            tag, out['IS'][0], out['VAL'][0], out['OOS'][0], out['OOS'][1], out['OOS'][2]))
    return out

print('\n=== ① 不剪枝：深度扫描（看 overfit） ===')
depth_scan = []
for dep in [1, 2, 3, 4, 6, 8, 12, None]:
    m = DecisionTreeRegressor(max_depth=dep, random_state=0)
    m.fit(tr[X], tr['y'])
    r = evaluate(m, 'max_depth=%s (叶子%d)' % (dep, m.get_n_leaves()), verbose=False)
    depth_scan.append((str(dep), r['IS'][0], r['VAL'][0], r['OOS'][0], m.get_n_leaves()))
    print('  depth=%-4s 叶子%6d | IS IC %+.4f | VAL %+.4f | OOS %+.4f' % (
        dep, m.get_n_leaves(), r['IS'][0], r['VAL'][0], r['OOS'][0]))

print('\n=== ② 代价复杂度剪枝（ccp_alpha，在验证集上选） ===')
base = DecisionTreeRegressor(max_depth=8, min_samples_leaf=200, random_state=0)
path = base.cost_complexity_pruning_path(tr[X], tr['y'])
alphas = np.unique(np.quantile(path.ccp_alphas, np.linspace(0.01, 0.95, 25)))
best = (None, -9)
rows = []
for a in alphas:
    m = DecisionTreeRegressor(max_depth=8, min_samples_leaf=200, ccp_alpha=a, random_state=0)
    m.fit(tr[X], tr['y'])
    ic_va = evaluate(m, '', verbose=False)['VAL'][0]
    rows.append((a, m.get_n_leaves(), ic_va))
    if ic_va > best[1]: best = (a, ic_va, m)
print('  验证集最优 alpha=%.5f（叶子 %d 个），验证 IC %+.4f' % (best[0], best[2].get_n_leaves(), best[1]))

print('\n=== ③ 对照模型 ===')
models = {}
m_pruned = best[2]; models['决策树(剪枝)'] = m_pruned
evaluate(m_pruned, '决策树 ccp_alpha=%.5f' % best[0])
m_d3 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=0).fit(tr[X], tr['y'])
models['决策树(depth=3)'] = m_d3
evaluate(m_d3, '决策树 max_depth=3')
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(tr[X], tr['y']); models['OLS'] = ols
evaluate(ols, '线性 OLS')
# 8因子等权 rank 合成
comp = df.copy()
for f in LIB:
    comp['r_'+f] = comp.groupby('date')[f].rank(pct=True)
comp['pred'] = comp[['r_'+f for f in LIB]].mean(axis=1)
for nm, d in [('IS', tr), ('VAL', va), ('OOS', te)]:
    dd = comp.loc[d.index]
    ic = daily_ic(dd, 'pred')
    if nm == 'OOS':
        print('  %-30s IS IC %+.4f | VAL %+.4f | OOS IC %+.4f (ICIR %+.2f, NW-t %+.1f)' % (
            '8因子 rank 合成', daily_ic(comp.loc[tr.index], 'pred').mean(),
            daily_ic(comp.loc[va.index], 'pred').mean(), ic.mean(), ic.mean()/ic.std(), nw_t(ic)))

print('\n=== ④ 剪枝后的树有多简单（前 3 层） ===')
rules = export_text(m_pruned, feature_names=X, max_depth=3, decimals=2)
print(rules)
print('特征重要性:', {k: round(v, 3) for k, v in zip(X, m_pruned.feature_importances_)})

print('\n=== ⑤ 分组收益（OOS，用剪枝树预测值分5组） ===')
te2 = te.copy(); te2['pred'] = m_pruned.predict(te[X])
te2['q'] = te2.groupby('date')['pred'].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
g = te2.groupby('q')['label'].mean() * 100
print(g.round(3).to_string(), '\n  多空(5日) %.3f%%' % (g.iloc[-1]-g.iloc[0]))

# ---- 图 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
d_lab = [d[0] for d in depth_scan]
ax[0].plot(d_lab, [d[1]*100 for d in depth_scan], 'o-', label='样本内 IC', color='#2980b9')
ax[0].plot(d_lab, [d[3]*100 for d in depth_scan], 's-', label='样本外 IC (2024+)', color='#c0392b')
ax[0].axhline(0, color='#7f8c8d', lw=.8)
ax[0].set_xlabel('max_depth'); ax[0].set_ylabel('Rank IC (%)')
ax[0].set_title('不剪枝：深度增加 → 样本内虚高、样本外崩溃', fontsize=10.5)
ax[0].legend(frameon=False); ax[0].grid(alpha=.25)
ax[1].bar(range(5), g.values, color=['#c0392b' if i in (0,4) else '#95a5a6' for i in range(5)])
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
ax[1].axhline(0, color='#7f8c8d', lw=.8)
ax[1].set_ylabel('5日平均收益 (%)')
ax[1].set_title('剪枝决策树 OOS 分组收益：多空 %.3f%%' % (g.iloc[-1]-g.iloc[0]), fontsize=10.5)
ax[1].grid(alpha=.25, axis='y')
plt.tight_layout(); plt.savefig(f'{ROOT}/reports/tree_pruning.png', dpi=135, bbox_inches='tight')
print('\n图已保存 reports/tree_pruning.png')

pd.DataFrame(depth_scan, columns=['max_depth', 'IS_IC', 'VAL_IC', 'OOS_IC', 'leaves']).to_csv(
    f'{ROOT}/reports/tree_depth_scan.csv', index=False)
