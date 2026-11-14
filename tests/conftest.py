# -*- coding: utf-8 -*-
"""pytest 公共 fixture

原则：逻辑测试用合成数据（毫秒级），契约测试才加载真实面板（十几秒，标 slow）。
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# 项目根目录：默认取本文件上一级，可用环境变量 QUANT_ROOT 覆盖（CI / 换机器不必改代码）
ROOT = os.environ.get('QUANT_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


# ---------------------------------------------------------------- 合成数据
@pytest.fixture
def toy_px():
    """6 天 × 2 只的迷你行情，第 4 天 A 停牌（close=0）"""
    dates = pd.date_range('2024-01-01', periods=6, freq='B')
    return pd.DataFrame({
        'date': list(dates) * 2,
        'symbol': ['000001'] * 6 + ['000002'] * 6,
        'close': [10.0, 10.5, 11.0, 0.0, 11.5, 12.0,      # A 第 4 天停牌
                  20.0, 20.2, 20.4, 20.6, 20.8, 21.0],
        'amount': [1e8] * 12,
        'turn': [2.0] * 12,
    })


@pytest.fixture
def toy_rets():
    """截面去均值前后的小矩阵：3 天 × 4 只"""
    return np.array([
        [0.01, 0.02, -0.01, 0.00],
        [0.05, -0.03, 0.02, np.nan],
        [-0.02, 0.01, 0.03, 0.04],
    ], dtype=np.float64)


@pytest.fixture
def toy_nav():
    """净值序列：个股跑赢基准 10%"""
    idx = pd.date_range('2024-01-01', periods=251, freq='B')
    s = pd.Series(np.linspace(1.0, 1.30, 251), index=idx)
    b = pd.Series(np.linspace(1.0, 1.18, 251), index=idx)
    return s, b


# ---------------------------------------------------------------- 真实数据（慢）
@pytest.fixture(scope='session')
def panel():
    p = f'{ROOT}/data/big/arr.npz'
    if not os.path.exists(p):
        pytest.skip('arr.npz 不存在')
    return np.load(p, allow_pickle=True)


@pytest.fixture(scope='session')
def daily_ret():
    p = f'{ROOT}/data/big/daily_ret.npz'
    if not os.path.exists(p):
        pytest.skip('daily_ret.npz 不存在')
    return np.load(p, allow_pickle=True)


@pytest.fixture(scope='session')
def panel_meta():
    """只读元信息，不碰 2.5 GB 的 X（用 Y2 的形状推，它只有 26 MB）"""
    p = f'{ROOT}/data/big/arr.npz'
    if not os.path.exists(p):
        pytest.skip('arr.npz 不存在')
    z = np.load(p, allow_pickle=True)
    y2 = z['Y2']
    return dict(
        shape=(y2.shape[0], y2.shape[1], len(z['feats'])),
        feats=list(z['feats']),
        dates=pd.to_datetime(pd.Series(z['dates']).astype(str)),
        symbols=[str(s).zfill(6) for s in z['symbols']],
        lock_start=str(z['lock_start']),
    )
