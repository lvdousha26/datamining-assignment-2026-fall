# -*- coding: utf-8 -*-
"""缠论因子增量检验：是「新结构」，还是 ret5 / dev20 的换皮？

三步：
  ① 与 42 个现有因子的截面秩相关（逐日均值）
  ② 对相关性最高的 2 个因子做截面回归，取残差，再算 IC（增量）
  ③ 分年度 IC，看稳定性

用法：python scripts/chan_check.py
产出：追加到 reports/chan_factors.md
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from wq101_factors import load_wide  # noqa: E402
import chan_factors as cf  # noqa: E402

ROOT = r'D:/quant'
N_SYM = 1500
TARGET = ['chan_bi_amp', 'chan_bi_speed', 'chan_zs_width', 'chan_zs_pos', 'chan_div']


def ic_series(f, Y2, days):
    out = {}
    for t in days:
        m = np.isfinite(f[t]) & np.isfinite(Y2[t])
        if m.sum() > 300:
            out[t] = spearmanr(f[t][m], Y2[t][m]).statistic
    return pd.Series(out).dropna()


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, Y2 = z['X'], z['Y2']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    syms_all = [str(s).zfill(6) for s in z['symbols']]
    feats = list(z['feats'])
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(syms_all, min(N_SYM, len(syms_all)), replace=False))
    ci = [syms_all.index(s) for s in pick]

    P = load_wide(pick)
    o, h, l, c = P['open'].values, P['high'].values, P['low'].values, P['close'].values
    pdate = pd.to_datetime(P['close'].index)

    F = {k: np.full_like(Y2, np.nan) for k in TARGET}
    for j, s in enumerate(pick):
        Fj = cf.one_stock(o[:, j], h[:, j], l[:, j], c[:, j])
        for k in TARGET:
            F[k][:, ci[j]] = pd.Series(Fj[k], index=pdate).reindex(dates).values
    print('缠论因子就绪（%d 只）' % len(pick), flush=True)

    tr = np.where((dates.dt.year >= 2016) & (dates.dt.year <= 2024))[0]

    # ---- ① 与 42 因子的相关 ----
    print('\n[① 与现有因子的截面秩相关]', flush=True)
    corr_tab = {}
    for k in TARGET:
        f = F[k]
        row = np.zeros(len(feats))
        for j in range(len(feats)):
            xs = []
            for t in tr[::3]:                       # 每 3 日抽样，省 2/3 时间
                m = np.isfinite(f[t]) & np.isfinite(X[t, :, j])
                if m.sum() > 300:
                    xs.append(spearmanr(f[t][m], X[t][m, j]).statistic)
            row[j] = np.nanmean(xs) if xs else np.nan
        corr_tab[k] = row
        top = np.argsort(-np.abs(row))[:3]
        print('  %-16s 最相关：%s' % (k, '、'.join(
            '%s %.2f' % (feats[i], row[i]) for i in top)), flush=True)
    pd.DataFrame(corr_tab, index=feats).round(3).to_csv(f'{ROOT}/reports/chan_corr.csv')

    # ---- ② 残差 IC ----
    print('\n[② 剔除最相关两因子后的残差 IC]', flush=True)
    res, yr_res, RES = [], {}, {}
    for k in TARGET:
        f = F[k]
        base = ic_series(f, Y2, tr)
        top = np.argsort(-np.abs(corr_tab[k]))[:2]
        resid = np.full_like(f, np.nan)
        for t in tr:
            m = np.isfinite(f[t]) & np.isfinite(Y2[t])
            for j in top:
                m &= np.isfinite(X[t, :, j])
            if m.sum() < 300:
                continue
            A = np.column_stack([np.ones(m.sum()), X[t, m, top[0]], X[t, m, top[1]]])
            y = f[t, m]
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid[t, m] = y - A @ beta
        ric = ic_series(resid, Y2, tr)
        yr_res[k] = pd.Series(ric.values, index=[dates[int(i)].year for i in ric.index]
                              ).groupby(level=0).mean()
        RES[k] = resid
        res.append(dict(factor=k, ic=base.mean(), icir=base.mean() / base.std(),
                        ctrl='+'.join(feats[j] for j in top),
                        resid_ic=ric.mean(), resid_icir=ric.mean() / ric.std() if ric.std() else np.nan,
                        keep=ric.mean() / base.mean() if base.mean() else np.nan))
        print('  %-16s IC %+.4f → 残差 %+.4f（保留 %3.0f%%，控制 %s）' % (
            k, base.mean(), ric.mean(), 100 * ric.mean() / base.mean() if base.mean() else 0,
            '+'.join(feats[j] for j in top)), flush=True)
    rs = pd.DataFrame(res)

    # ---- ③ 分年度 ----
    print('\n[③ 分年度 IC]', flush=True)
    yr = {}
    for k in TARGET:
        f = F[k]
        s = ic_series(f, Y2, tr)
        d = pd.Series(s.values, index=[dates[int(i)].year for i in s.index])
        yr[k] = d.groupby(level=0).mean()
    ydf = pd.DataFrame(yr)
    print(ydf.round(3).to_string(), flush=True)

    # ---- ④ 分位组合：IC 能不能变成钱 ----
    print('\n[④ 五分位组合（Y2 为 5 日超额，年化 = 均值×50）]', flush=True)
    qres = {}
    for k in ['chan_zs_pos', 'chan_div', 'chan_bi_speed']:
        f = RES[k]
        qs = [[] for _ in range(5)]
        for t in tr:
            m = np.isfinite(f[t]) & np.isfinite(Y2[t])
            if m.sum() < 300:
                continue
            v, y = f[t, m], Y2[t, m]
            q = pd.qcut(pd.Series(v).rank(method='first'), 5, labels=False).values
            for i in range(5):
                qs[i].append(y[q == i].mean())
        mu = [float(np.mean(x)) * 50 for x in qs]      # 5 日 → 年化
        qres[k] = mu
        print('  %-14s Q1 %+6.1f%%  Q2 %+6.1f%%  Q3 %+6.1f%%  Q4 %+6.1f%%  Q5 %+6.1f%%  Q5-Q1 %+6.1f%%' % (
            k, mu[0] * 100, mu[1] * 100, mu[2] * 100, mu[3] * 100, mu[4] * 100, (mu[4] - mu[0]) * 100), flush=True)

    # ---- 写报告 ----
    lines = ['## 附：增量的量化检验', '',
             '### ① 与 42 个现有因子的截面秩相关（2016-2024，逐日均值）', '',
             '| 缠论因子 | 最相关的现有因子 |', '|---|---|']
    for k in TARGET:
        row = corr_tab[k]
        top = np.argsort(-np.abs(row))[:3]
        lines.append('| `%s` | %s |' % (k, '、'.join('`%s` %.2f' % (feats[i], row[i]) for i in top)))
    lines += ['', '### ② 剔除最相关两因子后的残差 IC', '',
              '| 因子 | 原始 IC | 残差 IC | 保留比例 | 控制变量 |', '|---|---|---|---|---|']
    for _, r in rs.iterrows():
        lines.append('| `%s` | %+.4f | **%+.4f** | %.0f%% | %s |' % (
            r['factor'], r['ic'], r['resid_ic'], 100 * r['keep'], r['ctrl']))
    lines += ['', '### ③ 分年度 IC（判断稳定性）', '',
              '| 年份 | ' + ' | '.join('`%s`' % k.replace('chan_', '') for k in TARGET) + ' |',
              '|---|' + '---|' * len(TARGET)]
    for y, row in ydf.iterrows():
        lines.append('| %d | %s |' % (y, ' | '.join('%+.3f' % v for v in row.values)))
    rd = pd.DataFrame(yr_res)
    lines += ['', '**残差 IC 的分年度**（控制变量同上，检验增量是否只在个别年份成立）：', '',
              '| 年份 | ' + ' | '.join('`%s`' % k.replace('chan_', '') for k in TARGET) + ' |',
              '|---|' + '---|' * len(TARGET)]
    for y, row in rd.iterrows():
        lines.append('| %d | %s |' % (y, ' | '.join('%+.3f' % v for v in row.values)))
    rd.to_csv(f'{ROOT}/reports/chan_resid_yearly.csv')
    lines += ['', '### ④ 残差因子的五分位组合（Y2 年化，未计成本）', '',
              '| 因子 | Q1（最低） | Q2 | Q3 | Q4 | Q5（最高） | Q5−Q1 |', '|---|---|---|---|---|---|---|']
    for k, mu in qres.items():
        lines.append('| `%s` | %s | **%+.1f%%** |' % (k, ' | '.join('%+.1f%%' % (x * 100) for x in mu), (mu[4] - mu[0]) * 100))
    lines += ['']
    open(f'{ROOT}/reports/chan_check.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
    rs.to_csv(f'{ROOT}/reports/chan_check.csv', index=False)
    ydf.to_csv(f'{ROOT}/reports/chan_yearly.csv')
    print('\n→', f'{ROOT}/reports/chan_check.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='chan_check'):
        main()
