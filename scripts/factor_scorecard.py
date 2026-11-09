# -*- coding: utf-8 -*-
"""42 因子量化标准打分卡
检验项：
  ① 覆盖度  ② IC / ICIR / NW-t（含 Bonferroni 多重检验阈值）  ③ IC 胜率
  ④ 分年度符号稳定性  ⑤ 十分位单调性  ⑥ 多空毛收益
  ⑦ 换手率（top10% 每 5 日更替）  ⑧ 三层中性化 IC：行业 / 市值 / 行业+市值
评级：
  A 核心（行业+市值双中性后仍强）  B 可用  C 冗余/弱  D 无效
"""
import numpy as np, pandas as pd
from scipy import stats as sps

ROOT = r'D:/quant'
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates']); syms = np.array([str(s) for s in z['symbols']])
FEATS = list(z['feats']); T, N, F = X.shape
FI = {f: i for i, f in enumerate(FEATS)}

ind_map = pd.read_parquet(f'{ROOT}/data/big/ind_map.parquet')
im = dict(zip(ind_map['symbol'], ind_map['industry']))
IND = np.array([im.get(s, 'NA') for s in syms])
uind, IND_I = np.unique(IND, return_inverse=True)
SZ = X[:, :, FI['size']]
print('面板 %s | 行业 %d 类 | 未分类 %d 只' % (X.shape, len(uind), (IND == 'NA').sum()), flush=True)
print('注意：size 因子 = log(20日均成交额)，是流动性代理而非真实市值', flush=True)

CNT = np.zeros((10, F)); SUM = np.zeros((10, F))
ICS = {k: np.full((T, F), np.nan) for k in ['raw', 'ind', 'size', 'both']}
COV = np.zeros(F); TOP = [None] * F; TURN = [[] for _ in range(F)]

for t in range(T):
    m = M[t] & np.isfinite(Y[t])        # 必须排除 inf（停牌整日 close=0 造成的脏标签）
    n = int(m.sum())
    if n < 100: continue
    idx = np.where(m)[0]
    blk = X[t, idx, :].astype(np.float64)
    good = ~np.isnan(blk).any(0)
    COV += good
    yv = Y[t, idx].astype(np.float64)
    yc = (yv.argsort().argsort() + .5) / n - .5          # 标签 rank 中心化

    # ---- 三层中性化的因子矩阵 ----
    ii = IND_I[idx]
    cnt_i = np.bincount(ii, minlength=len(uind)).astype(float)
    s_ = np.zeros((len(uind), F)); np.add.at(s_, ii, blk)
    mu_i = s_ / np.maximum(cnt_i[:, None], 1)
    blk_ind = blk - mu_i[ii]                              # 去行业均值

    sz = np.nan_to_num(SZ[t, idx])
    ssz = np.zeros(len(uind)); np.add.at(ssz, ii, sz)
    sz_ind = sz - (ssz / np.maximum(cnt_i, 1))[ii]
    sz_n = sz_ind - sz_ind.mean()                         # size 也去行业均值+截面均值

    def resid_vs_size(B):
        b = (B * sz_n[:, None]).sum(0) / (sz_n @ sz_n + 1e-12)
        return B - sz_n[:, None] * b

    blk_size = resid_vs_size(blk - blk.mean(0))           # 仅去 size（含截距）
    blk_both = resid_vs_size(blk_ind)                     # 行业 + size

    def rank_ic(B):
        o = B.argsort(0); r = np.empty_like(o)
        np.put_along_axis(r, o, np.arange(n)[:, None].repeat(F, 1), 0)
        rc = (r + .5) / n - .5
        v = (rc * yc[:, None]).sum(0) / (n * rc.std(0) * yc.std() + 1e-12)
        v[~good] = np.nan
        return v, r

    for key, B in [('raw', blk), ('ind', blk_ind), ('size', blk_size), ('both', blk_both)]:
        ICS[key][t], rk = rank_ic(B)
        if key == 'raw':
            dec = (rk * 10 // n).astype(int).clip(0, 9)
            for k in range(F):
                if good[k]:
                    np.add.at(SUM[:, k], dec[:, k], yv)
                    np.add.at(CNT[:, k], dec[:, k], 1)
            if t % 5 == 0:
                for k in range(F):
                    if not good[k]: continue
                    top = set(idx[rk[:, k] >= n - max(n // 10, 5)])
                    if TOP[k] is not None and len(top):
                        TURN[k].append(1 - len(top & TOP[k]) / len(top))
                    TOP[k] = top

def nw_t(x, lag=5):
    x = np.asarray(x); x = x[~np.isnan(x)]
    n = len(x)
    if n < 10: return np.nan
    e = x - x.mean(); s = e @ e / n
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1); s += 2 * w * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan

rows = []
for k, f in enumerate(FEATS):
    d = {key: pd.Series(ICS[key][:, k], index=dates).dropna() for key in ICS}
    ic = d['raw']
    dec = SUM[:, k] / np.maximum(CNT[:, k], 1)
    mono = sps.spearmanr(np.arange(10), dec).correlation if CNT[:, k].min() > 0 else np.nan
    years = [ic[ic.index.year == y].mean() for y in range(2016, 2027)
             if len(ic[ic.index.year == y]) > 30]
    sign_match = sum(1 for v in years if np.sign(v) == np.sign(ic.mean()))
    rows.append(dict(factor=f, coverage=COV[k] / T,
                     IC=ic.mean(), ICIR=ic.mean() / ic.std(), t=nw_t(ic.values),
                     pos=(ic > 0).mean(), IC_24=ic[ic.index.year >= 2024].mean(),
                     sign_years='%d/%d' % (sign_match, len(years)),
                     mono=mono, LS=(dec[9] - dec[0]) * 100,
                     turnover=np.mean(TURN[k]) if TURN[k] else np.nan,
                     IC_ind=d['ind'].mean(), IC_size=d['size'].mean(), IC_both=d['both'].mean(),
                     IC_both_24=d['both'][d['both'].index.year >= 2024].mean()))
df = pd.DataFrame(rows)
for c in ['IC', 'IC_ind', 'IC_size', 'IC_both', 'IC_both_24']:
    df['a_' + c] = df[c].abs()

# 多重检验：42 因子 Bonferroni 5% 双侧 → |z| > 3.22（NW-t 已处理重叠标签自相关）
df['sig'] = df['t'].abs() > 3.22

def grade(r):
    if r['coverage'] < 0.6: return 'D'
    if r['a_IC_both'] >= 0.02 and r['a_IC_both_24'] >= 0.015 and r['sig']: return 'A'
    if r['a_IC_both'] >= 0.012 and r['sig']: return 'B'
    if r['a_IC'] >= 0.015 and r['sig']: return 'C'
    return 'D'
df['grade'] = df.apply(grade, axis=1)
# 市值依赖度：|IC| 被 size 中性化吃掉的比例
df['size_dep'] = 1 - df['a_IC_size'] / df['a_IC'].clip(lower=1e-6)

pd.set_option('display.width', 260)
cols = ['factor', 'grade', 'IC', 'ICIR', 't', 'pos', 'IC_24', 'sign_years', 'mono',
        'LS', 'turnover', 'IC_ind', 'IC_size', 'IC_both', 'IC_both_24', 'size_dep']
out = df.sort_values(['grade', 'a_IC_both'], ascending=[True, False])
print('\n=== 打分卡（按评级、双中性 |IC| 排序）===')
print(out[cols].round(3).to_string(index=False))

print('\n=== 汇总 ===')
print(df['grade'].value_counts().sort_index().to_string())
print('\n多重检验 Bonferroni(|t|>3.22) 通过 %d / %d' % (df['sig'].sum(), F))
print('原始 |IC|>0.02 : %d 个 | 双中性后 |IC|>0.02 : %d 个' %
      ((df['a_IC'] > .02).sum(), (df['a_IC_both'] > .02).sum()))
print('平均市值依赖度 %.0f%% | 高度依赖(>50%%) %d 个' %
      (df['size_dep'].mean() * 100, (df['size_dep'] > .5).sum()))
print('平均换手（top10%%/5日）%.0f%% | 中位 %.0f%%' %
      (df['turnover'].mean() * 100, df['turnover'].median() * 100))
print('\n市值依赖度最高的 8 个因子：')
print(df.nlargest(8, 'size_dep')[['factor', 'IC', 'IC_size', 'size_dep']].round(3).to_string(index=False))

df.to_csv(f'{ROOT}/reports/factor_scorecard.csv', index=False)
print('\n已存 reports/factor_scorecard.csv')
