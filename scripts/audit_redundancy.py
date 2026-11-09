# -*- coding: utf-8 -*-
"""冗余度审计：42 个显著因子到底等于几个独立 bet
① 因子值相关矩阵  ② PCA 独立维度  ③ IC 序列同步性  ④ Li&Ji 有效检验数
⑤ 逐日 IC 的 block bootstrap 置信区间（修正 NW-t 的乐观）
"""
import numpy as np, pandas as pd
from scipy import stats as sps

ROOT = r'D:/quant'
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates']); FEATS = list(z['feats'])
T, N, F = X.shape
pr = np.load(f'{ROOT}/data/big/preds.npz')
rng = np.random.default_rng(42)

# ---------- ①② 因子值相关 / PCA ----------
tt, ii = np.where(M & np.isfinite(Y))
k = rng.choice(len(tt), min(400000, len(tt)), replace=False)
S = X[tt[k], ii[k], :].astype(np.float64)
S = (S - S.mean(0)) / (S.std(0) + 1e-12)
C = np.corrcoef(S.T)
off = np.abs(C[np.triu_indices(F, 1)])
print('=== ① 因子相关（40 万行随机抽样）===')
print('  平均 |corr| %.3f | 中位 %.3f | 90分位 %.3f | >0.7 的配对 %d / %d' %
      (off.mean(), np.median(off), np.quantile(off, .9), (off > .7).sum(), len(off)))
pairs = [(abs(C[i, j]), FEATS[i], FEATS[j]) for i in range(F) for j in range(i + 1, F)]
print('  最相关的 6 对：', ', '.join('%s~%s %.2f' % (a, b, v) for v, a, b in sorted(pairs, reverse=True)[:6]))
eig = np.linalg.eigvalsh(C)[::-1]; cum = np.cumsum(eig) / eig.sum()
print('\n=== ② PCA 独立维度 ===')
print('  解释 80%% / 90%% / 95%% 方差需 %d / %d / %d 个成分（共 %d 因子）' %
      ((cum < .8).sum() + 1, (cum < .9).sum() + 1, (cum < .95).sum() + 1, F))
print('  前 5 特征值占比：', np.round(eig[:5] / eig.sum(), 3))

# ---------- ③ 逐日 IC 序列 ----------
print('\n=== ③ 逐日 IC 序列同步性 ===')
IC = np.full((T, F), np.nan)
for t in range(T):
    m = M[t] & np.isfinite(Y[t]); n = int(m.sum())
    if n < 100: continue
    idx = np.where(m)[0]
    blk = X[t, idx, :].astype(np.float32)
    o = blk.argsort(0); r = np.empty_like(o)
    np.put_along_axis(r, o, np.arange(n)[:, None].repeat(F, 1), 0)
    rc = (r + .5) / n - .5
    yv = Y[t, idx]; yc = (yv.argsort().argsort() + .5) / n - .5
    IC[t] = (rc * yc[:, None]).sum(0) / (n * rc.std(0) * yc.std() + 1e-12)
ICs = pd.DataFrame(IC, index=dates, columns=FEATS)
# 按 5 日非重叠取样，避免重叠标签自相关
ic5 = ICs.iloc[::5].dropna(how='all')
Ci = ic5.corr().values
oi = np.abs(Ci[np.triu_indices(F, 1)])
print('  IC 序列平均 |corr| %.3f（因子值 %.3f）→ 同涨同跌程度' % (oi.mean(), off.mean()))
e = np.linalg.eigvalsh(np.nan_to_num(Ci))[::-1]
print('  IC 序列第 1 主成分解释 %.1f%% 方差（前 3 合计 %.1f%%）' %
      (e[0] / e.sum() * 100, e[:3].sum() / e.sum() * 100))

# ---------- ④ Li & Ji 有效检验数 ----------
lam = np.abs(np.linalg.eigvalsh(np.nan_to_num(C)))
Meff = np.sum(np.where(lam >= 1, 1, lam - np.floor(lam)))
from scipy.stats import norm
thr42 = norm.ppf(1 - .05 / (2 * 42)); thrE = norm.ppf(1 - .05 / (2 * Meff))
print('\n=== ④ 多重检验修正 ===')
print('  名义检验数 42 | Li&Ji 有效检验数 %.1f' % Meff)
print('  Bonferroni |z| 阈值：按 42 → %.2f | 按 M_eff → %.2f' % (thr42, thrE))

# ---------- ⑤ block bootstrap ----------
def ic_series(p, t_, i_):
    d = pd.DataFrame({'d': t_, 'p': p, 'y': Y[t_, i_]})
    return d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()

print('\n=== ⑤ 逐日 IC 的 block bootstrap（20 交易日块，2000 次）===')
for nm, key in [('因子合成(线性)', 'fac'), ('Stacking', 'p_st')]:
    s = ic_series(pr[key + '_te'], pr['tt_te'], pr['ii_te']).values
    s = s[::5]                                    # 非重叠
    n = len(s); mu = s.mean(); se_naive = s.std() / np.sqrt(n)
    bs = []
    L = 20
    for _ in range(2000):
        st = rng.integers(0, n - L, size=int(np.ceil(n / L)))
        v = np.concatenate([s[a:a + L] for a in st])[:n]
        bs.append(v.mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    se_bs = np.std(bs)
    print('  %-16s IC %+.4f | naive SE %.4f (t=%.1f) | block SE %.4f (t=%.1f) | 95%%CI [%+.4f, %+.4f]' %
          (nm, mu, se_naive, mu / se_naive, se_bs, mu / se_bs, lo, hi))

ICs.iloc[::5].to_csv(f'{ROOT}/reports/ic_series_5d.csv')
print('\n已存 reports/ic_series_5d.csv')
