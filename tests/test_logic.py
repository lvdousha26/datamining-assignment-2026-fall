# -*- coding: utf-8 -*-
"""纯逻辑单测 —— 每个用例对应一个真实踩过的 bug

命名规则：test_<现象>_<期望>，docstring 写清「为什么必须有这条」。
这些不是为覆盖率写的，是为了「下次改代码时，别把这些坑再踩一遍」。
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


# ============================================================ 标签与清洗
def test_stop_price_not_minus_100(toy_px):
    """停牌股 close=0 必须先变 NaN，否则 fwd5 = 0/x-1 = −100% 假收益

    2026-11-09 在 short_d1.py 与 big_features.py 各踩过一次：
    第一版 D1 做空算到年化 −39.8%，好得不像真的，根因就是这个。
    """
    px = toy_px.copy()
    px['close'] = px['close'].where(px['close'] > 0)
    px['fwd5'] = px.groupby('symbol')['close'].transform(lambda s: s.shift(-5) / s - 1)
    v = px.loc[(px.symbol == '000001'), 'fwd5'].dropna()
    assert (v > -1).all(), '出现了 −100% 及以下的假收益'


def test_inf_does_not_spread():
    """单个 inf 会让 s - s.mean() 全变 inf，必须先去 inf 再去均值

    这是本项目最严重的一个 bug：13,505 行停牌股 close=0 → 未来收益分母 0 →
    截面去均值把 ±inf 扩散给当日全部股票，共 231,159 条（2.3%）。
    """
    s = pd.Series([0.01, 0.02, 0.03, np.inf, -0.01])
    bad = s - s.mean()                        # 错误做法
    assert not np.isfinite(bad).all(), '这个用例本身失效了'
    good = s.replace([np.inf, -np.inf], np.nan)
    good = good - good.mean()                 # 正确做法
    assert np.isfinite(good.dropna()).all()
    assert len(good.dropna()) == 4


def test_demean_sums_to_zero(toy_rets):
    """截面去均值后，当日有效样本之和应≈0"""
    for row in toy_rets:
        r = pd.Series(row).dropna()
        d = r - r.mean()
        assert abs(d.sum()) < 1e-12


def test_rank_transform_bounded(toy_rets):
    """训练标签用 (rank−0.5)/n，必须落在 (0,1) 且无 NaN"""
    r = pd.Series(toy_rets[0])
    rk = r.rank(pct=True)
    assert rk.min() > 0 and rk.max() <= 1
    assert rk.notna().all()


# ============================================================ 复权断层
def limit_of(symbols, dates):
    """A 股涨跌停幅度（规格来源：big_features.py 的断层反查逻辑）"""
    s = pd.Series(symbols, dtype=str)
    d = pd.to_datetime(pd.Series(dates))
    lim = np.full(len(s), 0.10)
    lim[s.str.startswith('688').values] = 0.20                      # 科创板
    cyb = s.str.startswith('30').values
    lim[cyb] = np.where(d[cyb].values >= np.datetime64('2020-08-24'), 0.20, 0.10)
    lim[s.str[0].isin(['8', '4']).values] = 0.30                    # 北交所
    return lim


def test_limit_rule():
    """涨跌停制度：主板 10%、创业板 2020-08-24 前后 10%/20%、科创 20%、北交 30%

    规则错了会把正常涨跌停误判成复权断层（第一版用 z-score 当收益，误判 92%）。
    """
    syms = ['600000', '300001', '300001', '688001', '830001']
    dts = ['2020-01-02', '2020-01-02', '2021-01-04', '2021-01-04', '2021-01-04']
    lim = limit_of(syms, dts)
    assert list(np.round(lim, 2)) == [0.10, 0.10, 0.20, 0.20, 0.30]


def test_cross_break_label_void():
    """窗口跨断层的 fwd5 必须作废——断层两侧价格不在同一复权尺度

    实测：断层后 20 日的 ret20 均值 +0.750，正常样本 +0.149，
    被系统性打上 0.7σ 的「假动量」。修完断层后 20 日有效样本 37,192 → 0。
    """
    px = pd.DataFrame({
        'symbol': ['A'] * 8,
        'close': [10, 10.1, 10.2, 20.0, 20.2, 20.4, 20.6, 20.8],   # 第 4 天翻倍=断层
    })
    px['ret1'] = px['close'].pct_change()
    px['brk'] = px['ret1'].abs() > 0.11
    assert px['brk'].iloc[3]

    px['fwd5'] = px['close'].shift(-5) / px['close'] - 1
    # t=2 的 fwd5 跨了 t=3 的断层
    cross = np.zeros(len(px), bool)
    for k in range(1, 6):
        cross |= px['brk'].shift(-k).fillna(False).astype(bool).values
    px.loc[cross, 'fwd5'] = np.nan
    assert np.isnan(px['fwd5'].iloc[2]), '跨断层的标签没有被作废'


# ============================================================ 事件研究
def test_car_geometric(toy_nav):
    """CAR 必须是几何超额 (1+r_s)/(1+r_b)−1，不是简单相减

    早期版本用 (1+超股)/(1+纳指)−1 是对的，但纯多头那段用了 abs = 超额 + 沪深300，
    基准错了（Y2 的基准是全市场等权），导致 A2 波动被低估一倍。
    """
    from event_study import car_at
    s, b = toy_nav
    c = car_at(s, b, 0, 250)
    r = s.iloc[250] / s.iloc[0] - 1
    rb = b.iloc[250] / b.iloc[0] - 1
    assert c == pytest.approx((1 + r) / (1 + rb) - 1, rel=1e-9)
    assert c != pytest.approx(r - rb, rel=1e-6), '退化成简单相减了'


def test_car_out_of_range(toy_nav):
    """窗口越界必须返回 nan，不能静默回绕或取最后一行"""
    from event_study import car_at
    s, b = toy_nav
    assert np.isnan(car_at(s, b, 250, 5))


def test_placebo_monotonic(toy_nav):
    """安慰剂分位应随实际 CAR 单调上升；且分位落在 [0,100]

    旧的手写脚本把 MSFT 的 95.6% 算成 52%，就是这里没测。
    """
    from event_study import placebo_pct
    s, b = toy_nav
    p_lo, _, n1 = placebo_pct(s, b, 60, -0.5)
    p_mid, _, _ = placebo_pct(s, b, 60, 0.0)
    p_hi, _, _ = placebo_pct(s, b, 60, 0.5)
    assert p_lo <= p_mid <= p_hi
    assert 0 <= p_lo <= 100 and 0 <= p_hi <= 100
    assert n1 > 30, '样本太少，安慰剂没有意义'


# ============================================================ 线性模型
def test_ridge_matches_ols_when_lambda_small():
    """lam→0 时 Ridge 必须退化成 OLS

    上一轮这里出过错：rhs 用了未标准化的 y（Aᵀy 而非 Aᵀy_z），
    系数被整体乘上 std(Y2)=0.068，看起来像「lam 一大就收缩 15 倍」，
    实际是自己算错了。
    """
    from stack_ridge import fit_ridge
    rng = np.random.default_rng(0)
    n, p = 500, 4
    A = np.column_stack([rng.normal(size=(n, p)), np.ones(n)])
    y = A @ np.array([0.5, -0.3, 0.2, 0.1, 1.0]) + rng.normal(scale=0.1, size=n)
    c_ridge = fit_ridge(A, y, 1e-12)
    c_ols = np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.allclose(c_ridge[:p + 1], c_ols, atol=1e-4)


def test_ridge_shrinks_with_lambda():
    """lam 增大时系数范数应单调不增（真收缩，不是尺度错）"""
    from stack_ridge import fit_ridge
    rng = np.random.default_rng(1)
    n, p = 300, 5
    A = np.column_stack([rng.normal(size=(n, p)), np.ones(n)])
    y = A @ rng.normal(size=p + 1) + rng.normal(scale=0.3, size=n)
    norms = [np.linalg.norm(fit_ridge(A, y, lam)[:p]) for lam in [1e-6, 1e-2, 1.0, 100.0]]
    assert all(norms[i] >= norms[i + 1] - 1e-9 for i in range(len(norms) - 1)), norms


# ============================================================ 工程细节
def test_weight_align_by_index():
    """权重必须按全局股票索引对齐，不能按位置相减

    strategy_compare.py 早期版本用 w2 - w1，逐日股票数不同直接 broadcast 报错；
    更隐蔽的情况是「不报错但错位」，会让换手算出 100%。
    """
    N = 5
    w1 = np.zeros(N); w1[[0, 2]] = 0.5
    w2 = np.zeros(N); w2[[0, 2, 4]] = 1 / 3
    to = np.abs(w2 - w1).sum() / 2
    assert 0 < to < 1
    # 按位置相减会得到什么（长度不同时直接崩，长度相同则错位）
    with pytest.raises(ValueError):
        _ = np.array([0.5, 0.5]) - np.array([0.33, 0.33, 0.34])


def test_runlog_creates_three_files(tmp_path, monkeypatch):
    """RunLogger 每次运行产出 .log / .json 并追加 index.csv"""
    from runlog import RunLogger
    with RunLogger('dummy_script.py', tag='test', echo=False, logdir=str(tmp_path)) as lg:
        print('hello')
    assert os.path.exists(lg.path)
    assert os.path.exists(lg.path.replace('.log', '.json'))
    assert os.path.exists(tmp_path / 'index.csv')
    body = open(lg.path, encoding='utf-8').read()
    assert 'status    : OK' in body
    assert 'hello' in body


def test_runlog_records_failure(tmp_path):
    """异常必须被记成 FAIL，不能记 OK"""
    from runlog import RunLogger
    p = None
    with pytest.raises(ValueError):
        with RunLogger('dummy.py', tag='fail', echo=False, logdir=str(tmp_path)) as lg:
            p = lg.path
            raise ValueError('boom')
    assert p and os.path.exists(p)
    assert 'FAIL(ValueError)' in open(p, encoding='utf-8').read()
