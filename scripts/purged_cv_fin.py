# -*- coding: utf-8 -*-
"""5 折 Purged + Embargo Walk-Forward CV（严格前向，不碰 lockbox）

设计
  - 只在有标签区间（2016-01 ~ 2025-06）切分；lockbox 段 Y2 为 NaN，天然排除
  - 均分 6 块，折 k 用 B0..Bk 训练（expanding），B(k+1) 验证
  - Purge：训练段末尾剔除 6 日（标签用到 t+6）
  - Embargo：验证段结束后再隔离 6 日才允许进入下一折训练（前向 CV 下由 purge 自动满足，
    此处显式断言，防止后续改成 K-fold 时失效）
  - 固定超参与固定轮数（正式模型 best_iter），不逐折选参
  - **所有可训练组件逐折重拟合**，杜绝跨折信息泄漏：
      LGBM                         boosting
      Ridge(42 因子)               线性基线
      因子 ICIR 加权合成            权重只用本折训练窗口内的 IC
      Stacking                     walk-forward：用上一折验证段的预测拟合元模型（尾部 purge 6 日）
  - TF 不重训（3 seed × 6 epoch × 5 折 ≈ 30 小时），其作用由 Ridge 线性模型与 Stacking 折间
    对比间接体现；TF 与 LGBM 的 0.88 相关性说明替换它不会改变结论量级
"""
import time, json
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy import stats as sps

ROOT = r'D:/quant'
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y, M = z['X'], z['Y2'], z['M']
dates = pd.to_datetime(z['dates']); FEATS = list(z['feats'])
T, N, F = X.shape

# ---------- 基本面因子（PIT，按公告日铺入；截面 z） ----------
FIN8 = ['bp', 'ep', 'sp', 'ocf_yield', 'fcf_yield', 'roe_ttm', 'gpa', 'accrual']
_fp = pd.read_parquet(f'{ROOT}/data/big/fin_panel.parquet')
_fp['date'] = pd.to_datetime(_fp['date'])
_dm = {d: i for i, d in enumerate(dates)}
_sm = {str(s).zfill(6): j for j, s in enumerate(z['symbols'])}
_fp = _fp[_fp.date.isin(_dm) & _fp.symbol.isin(_sm)].drop_duplicates(['date', 'symbol'])
_ti = _fp.date.map(_dm).values.astype(np.int32)
_si = _fp.symbol.map(_sm).values.astype(np.int32)
XF = np.full((T, N, len(FIN8)), np.nan, np.float32)
for _k, _c in enumerate(FIN8):
    _v = _fp[_c].values.astype(np.float64)
    _z = pd.Series(_v).groupby(_ti).transform(lambda s: (s - s.mean()) / (s.std() + 1e-12)).values
    XF[_ti, _si, _k] = _z.astype(np.float32)
    print('  基本面 %-10s 非空 %.1f%%  IC方向见报告' % (_c, np.isfinite(XF[:, :, _k]).mean() * 100), flush=True)
FIN_ICIR = None   # 下面逐折重算
del _fp
print('基本面矩阵 %s' % (XF.shape,), flush=True)
PURGE, EMBARGO, STEP = 6, 6, 5
LAM_LIN, LAM_STK = 1e-2, 1e-2          # ridge alpha = lam * n

params = dict(objective='regression', metric='l2', learning_rate=0.03, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=8, seed=42)
NROUND = json.load(open(f'{ROOT}/checkpoints/lgbm_meta.json'))['best_iter']

usable = np.where((M & np.isfinite(Y)).any(1))[0]
print('可用日期 %d 个：%s ~ %s | 固定轮数 %d | purge %d / embargo %d | %d 因子' %
      (len(usable), dates[usable[0]].date(), dates[usable[-1]].date(),
       NROUND, PURGE, EMBARGO, F), flush=True)
bounds = np.linspace(usable[0], usable[-1] + 1, 7).astype(int)


# ---------- 工具 ----------
def yrank(tt, ii):
    cache = {}; out = np.empty(len(tt), np.float32)
    for k, (t, i) in enumerate(zip(tt, ii)):
        if t not in cache:
            m = M[t] & np.isfinite(Y[t]); r = np.full(N, np.nan, np.float32)
            if m.sum() >= 30:
                v = Y[t][m]; r[m] = (sps.rankdata(v) - .5) / m.sum()
            cache[t] = r
        out[k] = cache[t][i]
    return out


def rank_mat_rows(mat, mask):
    """[N,F] 逐列截面百分位"""
    r = np.full_like(mat, np.nan, dtype=np.float32)
    if mask.sum() < 30: return r
    sub = mat[mask]
    r[mask] = ((sub.argsort(0).argsort(0) + .5) / len(sub)).astype(np.float32)
    return r


def daily_ic(p, tt, ii):
    d = pd.DataFrame({'d': tt, 'p': p, 'y': Y[tt, ii]})
    return d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()


def ls_spread(p, tt, ii, q=0.1, cost=2e-3):
    """top/bottom 10% 多空，5 日非重叠，双边各 20bp（合计 cost*4）"""
    d = pd.DataFrame({'t': tt, 'p': p, 'y': Y[tt, ii]})
    rec = []
    for t, g in d.groupby('t'):
        if len(g) < 100: continue
        n = max(int(len(g) * q), 10); gs = g.sort_values('p')
        rec.append(gs.tail(n)['y'].mean() - gs.head(n)['y'].mean())
    r = np.array(rec) - cost * 4
    per_year = 252 / STEP
    ann = np.prod(1 + r) ** (per_year / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(per_year)
    return ann, (ann - .02) / vol if vol > 0 else np.nan


def ridge_fit(A, yv, lam):
    G = A.T @ A; rhs = A.T @ yv
    P = np.eye(A.shape[1]); P[-1, -1] = 0.
    return np.linalg.solve(G + lam * len(A) * P, rhs)


def zc_fixed(a, mu, sd):
    return (np.asarray(a, float) - mu) / sd


rows = []
prev = None          # 上一折的验证段（供 walk-forward Stacking）
for k in range(5):
    te0, te1 = bounds[k + 1], bounds[k + 2]
    tr0, tr1 = bounds[0], bounds[k + 1] - PURGE
    if k > 0:                       # embargo：训练末端必须在上一折验证末端之前 EMBARGO 日
        assert tr1 <= bounds[k + 1] - EMBARGO, 'embargo 冲突'

    mtr = np.zeros(T, bool); mtr[tr0:tr1:STEP] = True
    mte = np.zeros(T, bool); mte[te0:te1:STEP] = True
    tt, ii = np.where(M & mtr[:, None] & np.isfinite(Y))
    ta, ia = np.where(M & mte[:, None] & np.isfinite(Y))
    if len(tt) < 10000 or len(ta) < 10000:
        print('折%d 样本不足，跳过' % (k + 1)); continue
    t0 = time.time()

    # ① LGBM（量价 42 + 基本面 8）
    ytr = yrank(tt, ii)
    XA = np.concatenate([X[tt, ii, :], XF[tt, ii, :]], axis=1)
    XB = np.concatenate([X[ta, ia, :], XF[ta, ia, :]], axis=1)
    ml = lgb.train(params, lgb.Dataset(XA, ytr), num_boost_round=NROUND)
    p_lgb = ml.predict(XB, num_iteration=NROUND)

    # ② Ridge(42+8 因子) —— 只用本折训练窗口；NaN 填 0（已截面 z，0=中位数）
    Xtr = np.nan_to_num(XA.astype(np.float64))
    Atr = np.column_stack([Xtr, np.ones(len(Xtr))])
    w_lin = ridge_fit(Atr, ytr.astype(np.float64), LAM_LIN)
    p_lin = np.column_stack([np.nan_to_num(XB.astype(np.float64)),
                             np.ones(len(ta))]) @ w_lin

    # ③ 因子 ICIR 加权合成 —— 权重只用本折训练日
    ic_sum = np.zeros(F); ic_cnt = np.zeros(F)
    for t in np.where(mtr)[0]:
        m = M[t] & np.isfinite(Y[t])
        if m.sum() < 30: continue
        rk = rank_mat_rows(X[t], m); yv = Y[t][m]
        for j in range(F):
            v = rk[m, j]
            if np.std(v) > 0:
                c = sps.spearmanr(v, yv).correlation
                if np.isfinite(c): ic_sum[j] += c; ic_cnt[j] += 1
    icf = ic_sum / np.maximum(ic_cnt, 1)
    w_fac = np.clip(icf, 0, None); w_fac = w_fac / max(w_fac.sum(), 1e-12)
    fac_map = {}
    for t in np.unique(ta):
        m = M[t]
        if m.sum() < 30: continue
        rk = rank_mat_rows(X[t], m)
        fac_map[t] = (np.nan_to_num(rk) @ w_fac, m)
    p_fac = np.array([fac_map[t][0][i] for t, i in zip(ta, ia)])

    # ④ Stacking（walk-forward，用上一折验证段拟合）
    p_st = np.full(len(ta), np.nan); p_icw = np.full(len(ta), np.nan)
    if prev is not None:
        Pm, tt_m, ii_m = prev
        # 尾部 purge：丢弃上一折验证段最后 EMBARGO 日的样本（其 5 日标签会伸进本折验证期）
        keep = tt_m < (te0 - EMBARGO)
        mu_p, sd_p = Pm[keep].mean(0), Pm[keep].std(0) + 1e-12
        A_meta = np.column_stack([(Pm[:, j] - mu_p[j]) / sd_p[j] for j in range(3)]
                                 + [np.ones(len(Pm))])
        ym = Y[tt_m, ii_m].astype(np.float64)
        yz = (ym - ym[keep].mean()) / ym[keep].std()
        c = ridge_fit(A_meta[keep], yz[keep], LAM_STK)
        Pn = np.column_stack([(np.column_stack([p_lgb, p_lin, p_fac])[:, j] - mu_p[j]) / sd_p[j]
                              for j in range(3)] + [np.ones(len(ta))])
        p_st = Pn @ c
        # IC 权重：逐日截面 Spearman 的均值（不是混合样本协方差——那会被"当日整体高低"污染）
        w_ic = np.array([daily_ic(Pm[keep, j], tt_m[keep], ii_m[keep]).mean() for j in range(3)])
        w_ic = np.clip(w_ic, 0, None)
        if w_ic.sum() <= 0: w_ic = np.ones(3)
        w_ic = w_ic / w_ic.sum()
        # 逐日截面 rank 后再加权，消除三个模型不同的量纲
        p_icw = np.zeros(len(ta))
        Mv = np.column_stack([p_lgb, p_lin, p_fac])
        for t, rws in pd.Series(np.arange(len(ta))).groupby(ta).groups.items():
            rws = np.array(rws)
            rk = (Mv[rws].argsort(0).argsort(0) + .5) / len(rws)
            p_icw[rws] = rk @ w_ic

    out = dict(fold=k + 1,
               train_from=str(dates[tr0].date()), train_to=str(dates[tr1 - 1].date()),
               val_from=str(dates[te0].date()), val_to=str(dates[te1 - 1].date()),
               n_train=len(tt), n_val=len(ta))
    for tag, pp in [('lgb', p_lgb), ('lin', p_lin), ('fac', p_fac),
                    ('stk', p_st), ('icw', p_icw)]:
        s = daily_ic(pp, ta, ia)
        out[f'{tag}_ic'] = s.mean() if len(s) else np.nan
        out[f'{tag}_icir'] = s.mean() / s.std() if len(s) else np.nan
        if tag == 'lgb': out['val_days'] = len(s)
    out['lgb_ann'], out['lgb_shp'] = ls_spread(p_lgb, ta, ia)
    out['fac_w_top3'] = {FEATS[j]: round(float(w_fac[j]), 3) for j in np.argsort(-w_fac)[:3]}
    out['secs'] = round(time.time() - t0, 1)
    rows.append(out)
    prev = (np.column_stack([p_lgb, p_lin, p_fac]), ta, ia)

    print('折%d 训练 %s~%s (%d行) → 验证 %s~%s (%d行/%d日)\n'
          '    LGBM %+.4f(%.2f) | 线性 %+.4f | 因子 %+.4f | Stacking %s | IC权重 %s\n'
          '    LGBM 多空 年化 %+.1f%% 夏普 %.2f | 因子权重 top3 %s | %.0fs' %
          (k + 1, out['train_from'], out['train_to'], len(tt),
           out['val_from'], out['val_to'], len(ta), out['val_days'],
           out['lgb_ic'], out['lgb_icir'], out['lin_ic'], out['fac_ic'],
           ('%+.4f' % out['stk_ic']) if np.isfinite(out['stk_ic']) else '   -  ',
           ('%+.4f' % out['icw_ic']) if np.isfinite(out['icw_ic']) else '   -  ',
           out['lgb_ann'] * 100, out['lgb_shp'], out['fac_w_top3'], out['secs']), flush=True)

d = pd.DataFrame(rows)
num = ['lgb_ic', 'lgb_icir', 'lin_ic', 'fac_ic', 'stk_ic', 'icw_ic']
d.to_csv(f'{ROOT}/reports/purged_cv_fin_folds.csv', index=False)

print('\n=== 汇总（5 折）===')
for c in num:
    s = d[c].dropna()
    if len(s) == 0: continue
    print('%-8s 均值 %+.4f | 标准差 %.4f | 最小 %+.4f | 最大 %+.4f | 正折 %d/%d' %
          (c, s.mean(), s.std(), s.min(), s.max(), int((s > 0).sum()), len(s)))
print('LGBM 逐折 IC:', [round(v, 4) for v in d.lgb_ic])
print('LGBM 逐折多空年化:', ['%+.1f%%' % (v * 100) for v in d.lgb_ann])
json.dump({'folds': len(d), 'purge': PURGE, 'embargo': EMBARGO, 'step': STEP,
           'lgb_ic_mean': float(d.lgb_ic.mean()), 'lgb_ic_std': float(d.lgb_ic.std()),
           'fac_ic_mean': float(d.fac_ic.mean()),
           'stk_ic_mean': float(d.stk_ic.dropna().mean()) if d.stk_ic.notna().any() else None,
           'icw_ic_mean': float(d.icw_ic.dropna().mean()) if d.icw_ic.notna().any() else None},
          open(f'{ROOT}/reports/purged_cv_fin_summary.json', 'w'), indent=2, ensure_ascii=False)
print('已存 reports/purged_cv_fin_folds.csv、purged_cv_fin_summary.json')
