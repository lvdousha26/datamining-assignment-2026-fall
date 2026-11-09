# -*- coding: utf-8 -*-
"""Stacking 元模型：OLS → Ridge 的等价性/增益检验

背景：big_train.py 的 Stacking 注释写 Ridge，实际用 np.linalg.lstsq（无 L2），
且 mat() 对每个分段各自 zc（系数在 meta 段拟合，却用验证段自己的均值方差去标准化）。
本脚本在**已冻结的 v2 preds** 上做三件事：

  1. 定位上一版"alpha 一大系数就掉 15 倍"的假象根因（rhs 用了未标准化的 y）
  2. 给出正确的 Ridge 系数-正则强度曲线（固定 meta 段的 mu/sd）
  3. 用 **meta 段内部时序切分**选 lam（不看 dev，避免在验证集上挑超参），
     再在 dev 上做一次确认

说明：lockbox 已消耗过（state runs=1），本脚本只读 preds.npz，不碰 lockbox 标签。
"""
import json
import numpy as np, pandas as pd
from scipy import stats as sps

ROOT = r'D:/quant'
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
Y2 = z['Y2']; dates = pd.to_datetime(z['dates'])
p = np.load(f'{ROOT}/data/big/preds.npz')
SEG = {k: (p[f'tt_{k}'], p[f'ii_{k}'],
           p[f'p_lgb_{k}'], p[f'p_tf_{k}'], p[f'fac_{k}']) for k in ['me', 'va', 'te']}

# ---------- 标准化口径 ----------
MU = np.array([SEG['me'][i].mean() for i in (2, 3, 4)])
SD = np.array([SEG['me'][i].std() for i in (2, 3, 4)])


def mat_fixed(s, cols=(2, 3, 4)):
    tt, ii = SEG[s][0], SEG[s][1]
    xs = [(SEG[s][c] - MU[c - 2]) / SD[c - 2] for c in cols]
    return np.column_stack(xs + [np.ones(len(tt))])


def mat_seg(s, cols=(2, 3, 4)):
    """旧口径：逐段各自 zc（会把"这一段整体偏高/偏低"也标准化掉）"""
    tt, ii = SEG[s][0], SEG[s][1]
    xs = []
    for c in cols:
        a = np.asarray(SEG[s][c], float)
        xs.append((a - a.mean()) / (a.std() + 1e-12))
    return np.column_stack(xs + [np.ones(len(tt))])


def ic(pred, k):
    tt, ii = SEG[k][0], SEG[k][1]
    d = pd.DataFrame({'d': tt, 'p': pred, 'y': Y2[tt, ii]})
    s = d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    return s.mean(), s.std(), len(s)


def fit_ridge(A, yv, alpha):
    """min ||yv - A b||^2 + alpha * ||b[:-1]||^2 ；截距不惩罚"""
    G = A.T @ A; rhs = A.T @ yv
    P = np.eye(A.shape[1]); P[-1, -1] = 0.0
    return np.linalg.solve(G + alpha * P, rhs)


# ---------- 1. 定位旧版系数异常 ----------
A = mat_seg('me'); tt, ii = SEG['me'][0], SEG['me'][1]
y = Y2[tt, ii].astype(np.float64)
yz = (y - y.mean()) / y.std()
n = len(A)
print('meta 段 %d 行 | Y2 均值 %+.6f 标准差 %.5f' % (n, y.mean(), y.std()), flush=True)

G = A.T @ A
print('G 对角 %s | 条件数 %.1f' % (np.round(np.diag(G), 0).astype(int), np.linalg.cond(G)), flush=True)

b_wrong = A.T @ y                    # ← 旧版：未标准化的 y
b_right = A.T @ yz                   # ← 修正
ols = np.linalg.lstsq(A, yz, rcond=None)[0]
print('\n[根因] rhs 用 y 而非 yz：系数被整体乘上 std(Y2)=%.4f，截距被塞进 mean(Y2)' % y.std())
for tag, bb in [('rhs=Aᵀy  (旧)', b_wrong), ('rhs=Aᵀyz (新)', b_right)]:
    co = np.linalg.solve(G + 11.48585 * np.diag([1, 1, 1, 0]), bb)
    print('  %-16s lam=1e-5 系数 %s' % (tag, np.round(co, 5)))
print('  %-16s 系数 %s' % ('OLS(lstsq)', np.round(ols, 5)))
print('  两版 rhs 之比 = %.4f，与 Y2 标准差 %.4f 一致 → 旧版"系数骤降 15 倍"是 rhs bug，'
      '不是 Ridge 在收缩' % (np.linalg.norm(b_wrong) / np.linalg.norm(b_right), y.std()))

# ---------- 2. 输入相关性 ----------
Z = mat_seg('me')[:, :3]
C = np.corrcoef(Z.T)
print('\nmeta 段三输入相关：LGBM-TF %.3f | LGBM-FAC %.3f | TF-FAC %.3f' %
      (C[0, 1], C[0, 2], C[1, 2]))

# ---------- 2b. OLS 权重的分块稳定性（解释为什么要 Ridge）----------
print('\n=== OLS 权重稳定性：meta 段按时间切 5 块分别拟合 ===')
ub = np.array_split(np.unique(tt), 5)
print('%-22s %9s %9s %9s' % ('块', 'LGBM', 'TF', 'fac'))
blk_coef = []
for k, days in enumerate(ub):
    m = np.isin(tt, days)
    c = np.linalg.lstsq(A[m], yz[m], rcond=None)[0]
    blk_coef.append(c[:3])
    print('块%d %s~%s %+9.4f %+9.4f %+9.4f' %
          (k + 1, dates[days[0]].date(), dates[days[-1]].date(), *c[:3]))
bc = np.array(blk_coef)
print('块间标准差 %s | fac 符号翻转 %s' %
      (np.round(bc.std(0), 4), '是' if (np.sign(bc[:, 2]).min() != np.sign(bc[:, 2]).max()) else '否'))

# ---------- 3. 正确的 Ridge 曲线（固定 meta mu/sd）----------
print('\n%-8s %12s %-38s %9s %9s %8s' %
      ('lambda', 'alpha', '系数 [LGBM, TF, fac, 截距]', 'meta IC', 'dev IC', 'dev 日数'))
GRID = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100, 300, 1000]
rows = []
for lam in GRID:
    alpha = lam * n
    coef = ols if lam == 0 else fit_ridge(A, yz, alpha)
    pred_me = A @ coef
    pred_te = mat_fixed('te') @ coef
    ic_me = ic(pred_me, 'me')[0]
    ic_te, _, nd = ic(pred_te, 'te')
    rows.append(dict(lam=lam, alpha=alpha, coef=np.round(coef, 5).tolist(),
                     ic_me=ic_me, ic_te=ic_te))
    print('%-8g %12.0f %-38s %+9.5f %+9.5f %8d' %
          (lam, alpha, str(np.round(coef, 4)), ic_me, ic_te, nd))
d = pd.DataFrame(rows)

# ---------- 4. meta 段内部时序选 lam（不看 dev）----------
print('\n=== 用 meta 段内部切分选 lam（前 60% 拟合 / 后 40% 验证，间隔 6 日 purge）===')
um = np.unique(tt)
cut = um[int(len(um) * 0.6)]
tr_m = np.where(tt < cut)[0]
va_m = np.where(tt >= cut)[0]
# purge：验证段起点前 6 个交易日内的训练样本剔除
keep = dates[tt[tr_m]] <= dates[cut - 6] if (cut - 6) >= 0 else np.ones(len(tr_m), bool)
tr_m = tr_m[keep]
Atr, ytr = A[tr_m], yz[tr_m]
Ava = A[va_m]
print('inner 训练 %s~%s (%d 行) → 验证 %s~%s (%d 行)' %
      (dates[tt[tr_m]].min().date(), dates[tt[tr_m]].max().date(), len(tr_m),
       dates[tt[va_m]].min().date(), dates[tt[va_m]].max().date(), len(va_m)))

inner = []
for lam in GRID:
    c = np.linalg.lstsq(Atr, ytr, rcond=None)[0] if lam == 0 else \
        fit_ridge(Atr, ytr, lam * len(Atr))
    pr = Ava @ c
    d_ = pd.DataFrame({'d': tt[va_m], 'p': pr, 'y': Y2[tt[va_m], ii[va_m]]})
    s = d_.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    inner.append(dict(lam=lam, ic_inner=s.mean(), n_days=len(s)))
    print('  lam %-6g inner IC %+.5f (%d 日)' % (lam, s.mean(), len(s)))
inn = pd.DataFrame(inner)
best = inn.loc[inn.ic_inner.idxmax()]
LAM = float(best.lam)
print('  → 选中 lam=%g（inner IC %+.5f）' % (LAM, best.ic_inner))

# ---------- 5. dev 确认 + 与旧 OLS / 2 输入版本对比 ----------
coef_ridge = ols if LAM == 0 else fit_ridge(A, yz, LAM * n)


def dev_ic(coef, cols=(2, 3, 4), seg='te'):
    return ic(mat_fixed(seg, cols) @ coef, seg)[0]


print('\n=== dev（2025H1）确认 ===')
print('  OLS  · 逐段 zc      IC %+.5f' % (ic(mat_seg('te') @ ols, 'te')[0]))
print('  OLS  · 固定 meta    IC %+.5f' % dev_ic(ols))
print('  Ridge(lam=%g)· 固定  IC %+.5f' % (LAM, dev_ic(coef_ridge)))
# 只用 LGBM+TF 的两输入版本
A2 = mat_fixed('me', (2, 3)); b2 = fit_ridge(A2, yz, LAM * len(A2))
print('  Ridge 两输入(LGBM+TF) IC %+.5f  ← fac 到底有没有用' % dev_ic(b2, (2, 3)))
print('  LGBM 单模型           IC %+.5f' % ic(SEG['te'][2], 'te')[0])
print('  TF  单模型            IC %+.5f' % ic(SEG['te'][3], 'te')[0])
print('  因子合成 单模型        IC %+.5f' % ic(SEG['te'][4], 'te')[0])

# ---------- 6. 部署口径：换 Ridge 后「禁买清单」变不变 ----------
print('\n=== 部署口径检查：垫底 10% 剔除（禁买名单）在 dev 段 ===')


def avoid(pred, k='te', q=0.1):
    """持有全市场 · 剔除当日垫底 q 的等权组合，相对全市场等权的 5 日超额（每 5 日取样，不重叠）"""
    tt, ii = SEG[k][0], SEG[k][1]
    d = pd.DataFrame({'t': tt, 'i': ii, 'p': pred, 'y': Y2[tt, ii]}).dropna()
    days = np.unique(d['t'])[::5]
    ex, ov = [], []
    prev_set = None
    for t in days:
        g = d[d['t'] == t]
        if len(g) < 100: continue
        n = max(int(len(g) * q), 10)
        bot = g.nsmallest(n, 'p')
        ex.append(g.drop(bot.index)['y'].mean() - g['y'].mean())
        cur = set(bot['i'].tolist())
        if prev_set is not None: ov.append(len(cur & prev_set) / n)
        prev_set = cur
    ex = np.array(ex)
    per_year = 252 / 5
    ann = np.prod(1 + ex) ** (per_year / len(ex)) - 1
    return ann, ex.mean(), (np.mean(ov) if ov else np.nan), len(ex)


for tag, pr in [('OLS stacking', mat_fixed('te') @ ols),
                ('Ridge lam=%g' % LAM, mat_fixed('te') @ coef_ridge),
                ('LGBM 单模型', SEG['te'][2]),
                ('TF 单模型', SEG['te'][3])]:
    ann, m, ov, nd = avoid(pr)
    print('  %-16s 剔底10%% 年化超额 %+6.2f%% | 单期均值 %+.4f | D1 换手 %.0f%% | %d 期' %
          (tag, ann * 100, m, ov * 100, nd))

p_ols = mat_fixed('te') @ ols; p_rid = mat_fixed('te') @ coef_ridge
tt, ii = SEG['te'][0], SEG['te'][1]
rc = []
for t in np.unique(tt):
    m = tt == t
    if m.sum() > 100:
        rc.append(sps.spearmanr(p_ols[m], p_rid[m]).correlation)
print('  OLS 与 Ridge 的逐日打分秩相关：均值 %.4f | 最低 %.4f' % (np.mean(rc), np.min(rc)))

out = dict(
    ridge_lambda=LAM, alpha=LAM * n, penalty='alpha*||b[:-1]||^2, 截距不惩罚',
    coef_order=['p_lgb', 'p_tf', 'fac', 'intercept'],
    coef=dict(zip(['p_lgb', 'p_tf', 'fac', 'intercept'], map(float, coef_ridge))),
    mu=MU.tolist(), sd=SD.tolist(),
    ic_dev=dict(ols_segzc=float(ic(mat_seg('te') @ ols, 'te')[0]),
                ols_fixed=float(dev_ic(ols)),
                ridge_fixed=float(dev_ic(coef_ridge)),
                ridge_2in=float(dev_ic(b2, (2, 3))),
                lgb=float(ic(SEG['te'][2], 'te')[0]),
                tf=float(ic(SEG['te'][3], 'te')[0]),
                fac=float(ic(SEG['te'][4], 'te')[0])),
    note='lam*len(A) 定义 alpha；旧版系数异常源于 rhs 用未标准化 y')
json.dump(out, open(f'{ROOT}/reports/ridge_config.json', 'w'), indent=2, ensure_ascii=False)
d.to_csv(f'{ROOT}/reports/ridge_check.csv', index=False)
inn.to_csv(f'{ROOT}/reports/ridge_inner.csv', index=False)
print('\n已存 reports/ridge_check.csv、ridge_inner.csv、ridge_config.json')
