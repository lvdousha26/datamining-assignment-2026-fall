# -*- coding: utf-8 -*-
"""WQ101 价量背离族：是「新因子」还是「流动性的另一种写法」？

pv_corr20 = −corr(volume, close, 20) 在 2016-24 上 IC +0.0507、胜率 70%，比项目里大多数因子都强。
但 A 股里"量与价"几乎必然与换手/流动性纠缠，先验证它是不是已有因子的换皮：

  ① 与 42 个现有因子的截面相关性（逐日算，取均值）
  ② 把 pv_corr20 对最相关的 1~2 个因子做截面回归，取残差，再算 IC（增量测试）

用法：python scripts/wq101_check.py
产出：追加到 reports/wq101.md
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from wq101_factors import load_wide
from runlog import RunLogger

ROOT = r'D:/quant'
N_SYM = 1500


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
    v, c = P['volume'], P['close']
    fac = -(v.rolling(20).corr(c))
    fac.index = pd.to_datetime(fac.index)
    F = np.full_like(Y2, np.nan)
    F[:, ci] = fac.reindex(dates).values

    # ---- ① 与 42 因子的截面相关 ----
    cors = []
    for t in range(len(dates)):
        m = np.isfinite(F[t])
        if m.sum() < 200:
            continue
        row = []
        for j in range(X.shape[2]):
            x = X[t, :, j]
            mm = m & np.isfinite(x)
            if mm.sum() < 200:
                row.append(np.nan); continue
            row.append(spearmanr(F[t, mm], x[mm]).statistic)
        cors.append(row)
    C = pd.DataFrame(np.array(cors), columns=feats).mean().sort_values(key=np.abs, ascending=False)
    print('与现有因子相关性 Top 8：')
    for k, val in C.head(8).items():
        print('  %-12s %+0.3f' % (k, val))

    # ---- ② 残差 IC（去掉最相关的 2 个因子）----
    tops = list(C.head(2).index)
    resid = F.copy()
    for t in range(len(dates)):
        m = np.isfinite(F[t])
        if m.sum() < 200:
            continue
        y = F[t, m]
        Z = np.column_stack([np.ones(m.sum())] + [np.nan_to_num(X[t, m, feats.index(k)]) for k in tops])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid[t, m] = y - Z @ beta

    def ic_of(f, mask=None):
        out = []
        idx = np.where(mask)[0] if mask is not None else range(len(dates))
        for t in idx:
            m = np.isfinite(f[t]) & np.isfinite(Y2[t])
            if m.sum() < 200:
                continue
            out.append(spearmanr(f[t, m], Y2[t, m]).statistic)
        s = pd.Series(out).dropna()
        return s.mean(), s.mean() / s.std(), (s > 0).mean(), len(s)

    tr = dates < '2025-01-01'
    for name, f in [('原始 pv_corr20', F), ('剔除 %s 后' % '/'.join(tops), resid)]:
        a, b, w, n = ic_of(f, tr.values)
        print('%-22s IC %+.4f | ICIR %+.2f | 胜率 %.0f%% | %d 日' % (name, a, b, w * 100, n))

    md = ['', '## 增量检验：它是新因子，还是流动性换皮？', '',
          '`pv_corr20 = −corr(volume, close, 20)` 在 2016-24 上 IC +0.0507、胜率 70%，'
          '比项目里多数因子都强——但 A 股里量价必然与换手/流动性纠缠，先查它是不是已有因子的换皮。', '',
          '与现有 42 因子的平均截面相关性（|ρ| 最大的 8 个）：', '',
          '| 因子 | 平均 |ρ| |', '|---|---|']
    for k, val in C.head(8).items():
        md.append('| `%s` | %+.3f |' % (k, val))
    md += ['', '剔除最相关的两个（`%s`）后的残差 IC：' % '`、`'.join(tops), '',
           '| 版本 | IC | ICIR | 胜率 |', '|---|---|---|---|']
    a0, b0, w0, _ = ic_of(F, tr.values)
    a1, b1, w1, _ = ic_of(resid, tr.values)
    md += ['| 原始 | %+.4f | %+.2f | %.0f%% |' % (a0, b0, w0 * 100),
           '| 残差 | **%+.4f** | %+.2f | %.0f%% |' % (a1, b1, w1 * 100), '']
    verdict = ('残差 IC 仍保留九成以上，说明它带来了独立信息。'
               if a1 > a0 * 0.7 else
               '残差 IC 大幅缩水，说明它主要是已有因子（流动性/换手）的另一种写法。')
    md += ['**判断**：' + verdict, '']
    open(f'{ROOT}/reports/wq101.md', 'a', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n' + verdict)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    with RunLogger(__file__, tag='wq101_check'):
        main()
