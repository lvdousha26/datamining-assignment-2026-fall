# -*- coding: utf-8 -*-
"""Stacking 元模型：真正的 Ridge（含标准化口径与正则强度选择）

修复两个历史问题：
  1) big_train.py 的 Stacking 注释写 Ridge，实际是 np.linalg.lstsq（无 L2）；
     三个输入（LGBM/TF/因子）两两相关 0.39~0.88，OLS 权重极不稳定
     （meta 段切 5 块分别拟合，LGBM 权重在 -0.06~+0.22 之间摆动、因子符号翻转）。
  2) mat() 对每个分段各自 zc，而系数在 meta 段拟合 —— 等于让验证段用自己的均值方差
     重新标定输入，截距与斜率口径不一致。本模块固定 meta 段的 mu/sd。

约定
  - 目标函数：min ||y - A b||^2 + alpha * ||b[:-1]||^2，截距不惩罚
  - alpha = lam * n（n 为样本数），保证 lam 是"相对收缩强度"，与样本量无关
  - lam 用**段内时序切分**选取（前 60% 拟合 / 后 40% 验证，间隔 6 日 purge），不看验证集
"""
import numpy as np
from scipy import stats as sps


def stats_of(*preds):
    """返回用于标准化的 mu / sd（在 meta 段调用一次，之后固定）"""
    mu = np.array([np.asarray(p, float).mean() for p in preds])
    sd = np.array([np.asarray(p, float).std() for p in preds]) + 1e-12
    return mu, sd


def design(preds, mu, sd):
    """[z(p1), z(p2), ..., 1]，标准化参数固定传入"""
    cols = [(np.asarray(p, float) - mu[j]) / sd[j] for j, p in enumerate(preds)]
    cols.append(np.ones(len(cols[0])))
    return np.column_stack(cols)


def fit_ridge(A, y, lam):
    """闭式解；lam=0 时等价 OLS（仍用正规方程，np.linalg.solve 自带最小二乘解）"""
    G = A.T @ A
    rhs = A.T @ y
    if lam <= 0:
        return np.linalg.lstsq(A, y, rcond=None)[0]
    P = np.eye(A.shape[1]); P[-1, -1] = 0.0
    return np.linalg.solve(G + lam * len(A) * P, rhs)


def select_lambda(A, y, tt, grid=(0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3), frac=0.6, purge=6):
    """段内时序切分选 lam：返回 (lam, 明细 DataFrame)。评价指标为逐日 Rank IC。"""
    import pandas as pd
    days = np.unique(tt)
    cut = days[int(len(days) * frac)]
    tr = np.where(tt < cut)[0]
    va = np.where(tt >= cut)[0]
    if purge > 0:                                  # 训练尾部再隔离 purge 日
        tr = tr[tt[tr] < cut - purge]
    rows = []
    for lam in grid:
        c = fit_ridge(A[tr], y[tr], lam)
        p = A[va] @ c
        d = pd.DataFrame({'d': tt[va], 'p': p, 'y': y[va]})
        s = d.groupby('d').apply(
            lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
        rows.append(dict(lam=lam, ic=s.mean() if len(s) else np.nan))
    t = pd.DataFrame(rows)
    return float(t.loc[t.ic.idxmax(), 'lam']), t


def fix_scale(coef, A, target_std=1.0):
    """把系数整体缩放到"预测在这些样本上 std=target_std"，只影响数值尺度不影响排序。
    强正则下 lam 越大系数越小（1e-4 量级），缩放后下游读取更安全。"""
    s = (A @ coef).std()
    return coef * (target_std / s) if s > 0 else coef


def stack(preds_meta, y_meta, tt_meta, preds_apply, y_apply=None,
          grid=(0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3), lam=None):
    """一步完成：固定标准化 → 选 lam（或指定）→ 拟合 → 应用到新段

    preds_meta : (p1, p2, p3) 均为 meta 段预测（等长）
    preds_apply: 同上，待打分段
    返回 (score, info)
    """
    mu, sd = stats_of(*preds_meta)
    A = design(preds_meta, mu, sd)
    y = np.asarray(y_meta, float)
    y = (y - y.mean()) / y.std()
    if lam is None:
        lam, tab = select_lambda(A, y, tt_meta, grid=grid)
    else:
        tab = None
    coef = fit_ridge(A, y, lam)
    coef_s = fix_scale(coef, A)
    score = design(preds_apply, mu, sd) @ coef_s
    return score, dict(lam=lam, coef=coef_s, coef_raw=coef, mu=mu, sd=sd, table=tab)


if __name__ == '__main__':
    # 自检：在冻结的 v2 preds 上复现 reports/ridge_check.* 的结论
    import pandas as pd
    ROOT = r'D:/quant'
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    Y2 = z['Y2']
    p = np.load(f'{ROOT}/data/big/preds.npz')
    seg = {k: (p[f'tt_{k}'], p[f'ii_{k}'], p[f'p_lgb_{k}'], p[f'p_tf_{k}'], p[f'fac_{k}'])
           for k in ['me', 'va', 'te']}
    me, te = seg['me'], seg['te']
    s, info = stack((me[2], me[3], me[4]), Y2[me[0], me[1]], me[0],
                    (te[2], te[3], te[4]))
    ols_s, _ = stack((me[2], me[3], me[4]), Y2[me[0], me[1]], me[0],
                     (te[2], te[3], te[4]), lam=0)
    for tag, sc in [('OLS', ols_s), ('Ridge', s)]:
        d = pd.DataFrame({'d': te[0], 'p': sc, 'y': Y2[te[0], te[1]]})
        ic = d.groupby('d').apply(
            lambda g: sps.spearmanr(g['p'], g['y']).correlation).mean()
        print('%-6s lam=%-8g dev IC %+.5f' % (tag, 0 if tag == 'OLS' else info['lam'], ic))
