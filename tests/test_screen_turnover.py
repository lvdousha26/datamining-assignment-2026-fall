# -*- coding: utf-8 -*-
"""禁买清单的名单更替率（对应 2026-11-09 修的真 bug）

bug 现场
--------
`evaluate_screen.py` 原本这样算 D1 名单更替：

    cur = set(bot.index)          # bot = 当日打分垫底 10% 的 DataFrame

`bot.index` 是 **DataFrame 行索引**，而不同交易日的行天然是不同行，索引不可能相同。
于是 `len(cur - prev)` 每期都等于 n，更替率恒为 100%。

后果与症状：
  - `reports/screen_eval.json` 里出现 `turnover = 0.979`（= 47/48，首期记 0，其余 47 期都是 1.0）
  - 成本项 `to * q * 2 * 15bp` 从 1.34bp/期 虚增到 2.94bp/期
  - 年化超额被低估约 0.8pp（2024 +4.99% → 真实 +5.85%，2025H1 +3.83% → +4.67%）

修法：用股票在面板里的列索引 `i`（跨日稳定），并把逻辑收进 `screen_stats.d1_churn`。
这三个用例分别盯住：算法本身、失败模式、以及"脚本有没有偷懒绕开这个函数"。
"""
import os

import numpy as np
import pandas as pd
import pytest

from screen_stats import d1_churn, mean_churn


def _day(t, bottom_ids, n_total=250):
    """构造一天：指定 id 集合拿到最低分，其余给一个明显更大的常数"""
    ids = np.arange(n_total)
    p = np.full(n_total, 5000.0)
    p[list(bottom_ids)] = np.arange(len(bottom_ids), dtype=float)
    return pd.DataFrame({'t': t, 'i': ids, 'p': p})


def test_churn_is_name_based_not_row_based():
    """两天 D1 重叠 13/25 → 更替率必须是 0.48，不是 1.0

    用行索引算会得到 1.0——这正是那个 bug 的指纹。
    """
    d0 = _day(0, range(0, 25))            # D1 = 0..24
    d1 = _day(1, range(12, 37))           # D1 = 12..36，重叠 12..24 共 13 只
    rows = d1_churn(pd.concat([d0, d1], ignore_index=True), step=1)  # 每天都取样

    assert len(rows) == 2
    t0, n0, c0 = rows[0]
    t1, n1, c1 = rows[1]
    assert n0 == n1 == 25
    assert c0 == 0.0, '首期没有"上一期"，更替率记 0'
    assert c1 == pytest.approx(12 / 25), '新进 12 只（25..36）/ 25'
    assert c1 < 1.0, '恒为 1.0 说明又在用行索引'


def test_identical_list_has_zero_churn():
    """两天一模一样 → 0% 更替（行索引版本会算成 100%，这是最刺眼的反例）"""
    rows = d1_churn(pd.concat([_day(0, range(25)), _day(1, range(25))], ignore_index=True), step=1)
    assert rows[1][2] == 0.0
    assert mean_churn(rows) == 0.0


def test_missing_symbol_index_raises():
    """没有 `i` 列时必须报错，而不是悄悄退回行索引"""
    df = pd.DataFrame({'t': [0] * 250, 'p': np.arange(250, dtype=float)})
    with pytest.raises(KeyError, match="股票索引"):
        d1_churn(df)


def test_evaluate_screen_goes_through_d1_churn():
    """脚本必须调用这个函数，不能自己再写一遍 set(...index)"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, 'scripts', 'evaluate_screen.py'), encoding='utf-8').read()
    assert 'from screen_stats import d1_churn' in src
    assert 'set(bot.index)' not in src, '2026-11-09 的 bug 又回来了'
    assert 'set(b.index)' not in src, '分年那段的老写法又回来了'
