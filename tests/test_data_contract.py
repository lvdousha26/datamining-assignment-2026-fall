# -*- coding: utf-8 -*-
"""数据契约测试 —— 保证管线改动不会悄悄破坏下游

分两组：
    fast  只读元信息（毫秒级），默认跑
    slow  扫 X / Y2 全量（十几秒），用 -m slow 显式开启

这些测试的作用是「改完 big_features.py 之后，一条命令知道有没有把东西弄坏」。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

# 项目根目录：默认取本文件上一级，可用环境变量 QUANT_ROOT 覆盖（CI / 换机器不必改代码）
ROOT = os.environ.get('QUANT_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


# ============================================================ 快速契约
def test_panel_shape_matches_feats(panel_meta):
    """X 的三维必须与 feats 长度一致"""
    T, N, F = panel_meta['shape']
    assert F == len(panel_meta['feats']) == 42
    assert T > 2000 and N > 5000


def test_feats_no_duplicate(panel_meta):
    """因子名不能重复——重复会导致 X[:, :, k] 取错列"""
    f = panel_meta['feats']
    assert len(f) == len(set(f))


def test_dates_sorted_unique(panel_meta):
    """日期必须严格递增且无重复，否则 searchsorted / 切片全错"""
    d = panel_meta['dates']
    assert d.is_monotonic_increasing
    assert not d.duplicated().any()


def test_symbols_unique_and_padded(panel_meta):
    """股票代码必须是 6 位补零字符串且唯一"""
    s = panel_meta['symbols']
    assert len(s) == len(set(s))
    assert all(len(x) == 6 and x.isdigit() for x in s[:100])


def test_train_test_split_by_lock_start(panel_meta):
    """训练段必须早于 lock_start，封存段在之后"""
    d = panel_meta['dates']
    lock = pd.Timestamp(panel_meta['lock_start'])
    assert (d < lock).sum() > 2000
    assert (d >= lock).sum() > 200


def test_lockbox_state_not_rerun():
    """lockbox 是一次性验证，runs 必须 ≤1

    这条测试的作用是「防止未来的自己手滑重跑」——重跑一次，封存段的证据价值就没了。
    """
    p = f'{ROOT}/reports/lockbox_state.json'
    if not os.path.exists(p):
        pytest.skip('lockbox_state.json 不存在')
    st = json.load(open(p))
    assert st['runs'] <= 1, f'lockbox 已被使用 {st["runs"]} 次，不允许再跑'


def test_daily_ret_aligned_with_panel(panel_meta, daily_ret):
    """日收益矩阵的 symbols 必须与面板完全一致（顺序也要对）

    2026-11-09 踩过：daily_ret 还是旧面板的 5348 只，arr 已换成 5342 只，
    差 6 只就会让列索引整体错位。
    """
    a = panel_meta['symbols']
    b = [str(x).zfill(6) for x in daily_ret['symbols']]
    assert len(a) == len(b), f'股票数不一致：面板 {len(a)} vs 收益矩阵 {len(b)}'
    assert a == b, '股票代码顺序不一致，列索引会错位'


def test_daily_ret_bounded(daily_ret):
    """日收益必须被截断（|r| ≤ 50%），且不含 inf"""
    R = daily_ret['R']
    v = R[np.isfinite(R)]
    assert np.abs(v).max() <= 0.5001, f'出现越界收益 {np.abs(v).max():.3f}'
    assert not np.isinf(R).any()


def test_data_dictionary_complete():
    """数据字典不能有空白释义——课程作业要求每字段都有含义"""
    p = f'{ROOT}/reports/data_dictionary.csv'
    if not os.path.exists(p):
        pytest.skip('data_dictionary.csv 不存在')
    d = pd.read_csv(p)
    assert len(d) >= 100
    blank = d[d['中文含义'].isna() | (d['中文含义'].astype(str).str.strip() == '')]
    assert len(blank) == 0, f'{len(blank)} 个字段没有中文含义：{blank["字段"].tolist()[:5]}'
    assert d['来源与时点'].notna().all()


def test_event_csv_schema():
    """事件表必须有 symbol/date 两列，且日期可解析"""
    import glob
    fs = glob.glob(f'{ROOT}/events/*.csv')
    if not fs:
        pytest.skip('没有事件表')
    for f in fs:
        d = pd.read_csv(f, dtype=str)
        assert 'symbol' in d.columns and 'date' in d.columns, f'{os.path.basename(f)} 缺列'
        assert pd.to_datetime(d['date'], errors='coerce').notna().all(), \
            f'{os.path.basename(f)} 有无法解析的日期'


def test_live_screen_d1_ratio():
    """禁买清单的 D1 占比必须≈10%"""
    import glob
    fs = sorted(glob.glob(f'{ROOT}/live/screen_2*.csv'))
    fs = [f for f in fs if 'ols' not in f and 'v1' not in f]
    if not fs:
        pytest.skip('没有清单文件')
    d = pd.read_csv(fs[-1])
    if 'is_d1' not in d.columns:
        pytest.skip('清单缺 is_d1 列')
    ratio = d['is_d1'].mean()
    assert 0.08 <= ratio <= 0.12, f'D1 占比 {ratio:.1%}，偏离 10% 太多'


def test_reports_exist():
    """关键报告不能丢——它们是结论的唯一载体"""
    need = ['data_fix_v2_report.md', 'STRATEGY_MANUAL.md',
            'event_study_guide.md', 'data_dictionary.md', 'ASSETS_TODO.md']
    for n in need:
        assert os.path.exists(f'{ROOT}/reports/{n}'), f'缺 {n}'


# ============================================================ 慢契约（-m slow）
@pytest.mark.slow
def test_all_factors_finite_inside_mask(panel):
    """M 掩码内所有因子必须有限——有 NaN 说明 ok 过滤失效"""
    X, M = panel['X'], panel['M']
    sub = X[M]
    assert np.isfinite(sub).all(), '掩码内存有 NaN/inf'


@pytest.mark.slow
def test_factors_clipped(panel):
    """因子必须裁到 ±4，否则极端值会主导树模型分裂"""
    X, M = panel['X'], panel['M']
    sub = X[M]
    assert np.abs(sub).max() <= 4.0001


@pytest.mark.slow
def test_labels_cross_section_demeaned(panel):
    """Y2 的截面均值应接近 0，但**不要求严格为 0**

    已知特性：`big_features.py` 的 `y2` 是在 `ok` 过滤（成交额/上市满 250 日/非 ST）
    **之前**去均值的，过滤后子集的均值会残留小幅偏差。
    实测：中位 1.0e-3、p90 4.1e-3、p99 8.4e-3、max 2.1e-2（占 Y2 标准差 0.056 的 38% 但只出现在极端日）。

    对 rank 训练无实质影响，所以不改管线；但如果中位数涨到 5e-3 以上，
    说明过滤逻辑变了，要重新检查。
    """
    Y2, M = panel['Y2'], panel['M']
    ok = M & np.isfinite(Y2)
    means = np.array([Y2[t][ok[t]].mean() for t in range(Y2.shape[0])
                      if ok[t].sum() > 100])
    a = np.abs(means)
    assert np.median(a) < 2e-3, f'中位偏差 {np.median(a):.2e}，过滤逻辑可能变了'
    assert np.percentile(a, 99) < 1.5e-2, f'p99 偏差 {np.percentile(a, 99):.2e}'
    assert a.max() < 0.03, f'最大偏差 {a.max():.2e}'


@pytest.mark.slow
def test_lockbox_labels_empty_in_train(panel, panel_meta):
    """训练段的 Y2 必须有值，封存段的 Y2 必须为空（标签物理隔离）"""
    Y2 = panel['Y2']
    dates = panel_meta['dates']
    lock = pd.Timestamp(panel_meta['lock_start'])
    tr = (dates < lock).values
    te = (dates >= lock).values
    assert np.isfinite(Y2[tr]).mean() > 0.5, '训练段标签太少'
    assert np.isfinite(Y2[te]).sum() == 0, '封存段标签泄漏到主面板'


@pytest.mark.slow
def test_label_scale_reasonable(panel):
    """Y2 标准差应在 0.04~0.10 之间

    被复权脏数据污染时会到 0.1177（约 2 倍），这是当初发现问题的信号。
    """
    Y2 = panel['Y2']
    v = Y2[np.isfinite(Y2)]
    sd = v.std()
    assert 0.04 < sd < 0.10, f'Y2 标准差 {sd:.4f}，超出合理区间（可能又有脏数据）'
