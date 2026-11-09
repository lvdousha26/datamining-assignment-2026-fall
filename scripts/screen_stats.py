# -*- coding: utf-8 -*-
"""禁买清单的截面统计小工具（可单独测试，不依赖 2.5 GB 面板）

存在的理由：2026-11-09 发现 `evaluate_screen.py` 用 **DataFrame 行索引**算 D1 名单更替，
而不同交易日的行索引天然不重叠，于是每期都被算成「100% 更替」
（`screen_eval.json` 里那个 97.9% = 47/48，首期记 0），成本被多扣约 1.7bp/期。
真实更替率只有 44%。

所以把这段逻辑收进一个函数、只认股票索引 `i`，并让测试直接盯住它。
"""
import numpy as np


def d1_churn(df, q=0.10, step=5, min_n=200):
    """D1（打分垫底 q）名单的逐期更替率

    参数
    ----
    df : DataFrame，必须含 `t`（交易日索引）、`i`（**股票列索引**，跨日稳定）、`p`（打分）
         注意：不要传行索引——行索引每天不同，会把每期都算成 100% 更替。
    q  : 剔除比例
    step : 每多少个交易日取一次

    返回
    ----
    list of (t, n, churn)；churn = 本期新进 D1 的只数 / n，首期为 0.0
    """
    if 'i' not in df.columns:
        raise KeyError("d1_churn 需要股票索引列 'i'（不能用行索引：跨日不重叠会算出 100% 更替）")

    days = np.sort(df['t'].unique())[::step]
    out, prev = [], None
    for t in days:
        g = df[df['t'] == t]
        if len(g) < min_n:
            continue
        n = max(int(len(g) * q), 10)
        cur = set(g.nsmallest(n, 'p')['i'])
        churn = 0.0 if prev is None else len(cur - prev) / n
        out.append((t, n, churn))
        prev = cur
    return out


def mean_churn(churn_rows):
    """把 d1_churn 的返回值压成一个数（首期记 0 会稀释均值，需要时自己剔掉）"""
    vals = [c for _, _, c in churn_rows[1:]]
    return float(np.mean(vals)) if vals else 0.0
