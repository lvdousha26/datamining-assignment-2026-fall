# -*- coding: utf-8 -*-
"""生成数据字典（字段名/含义/类型/单位/取值范围/缺失率/来源与时点）

输出
    reports/data_dictionary.csv    机器可读，每行一个字段
    reports/data_dictionary.md     人读版，按数据集分节

中文含义与单位靠下面的 META 手工维护；统计量（缺失率、范围、样例）从真实数据扫。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

# ---- 运行日志：统一写 logs/，不改变原有输出 ----
if __name__ == '__main__':
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from runlog import RunLogger as _RL
        _RL(__file__).__enter__()
    except Exception:
        pass

ROOT = r'D:/quant'
BIG = f'{ROOT}/data/big'

# ---------------------------------------------------------------- 字段释义
# (数据集, 字段) -> (中文含义, 单位, 来源与时点)
META = {
    # ---- 行情面板 ----
    'panel.date': ('交易日', '—', '东方财富日线（后复权）'),
    'panel.symbol': ('股票代码（6 位）', '—', '东方财富，含已退市'),
    'panel.y': ('未来 5 日收益（t 收→t+5 收），截面去均值', '小数', '由 hfq 收盘价计算'),
    'panel.y_rk': ('y 的当日截面分位（0~1）', '分位', '同上'),
    'panel.fwd5': ('未来 5 日原始收益（未去均值）', '小数', '同上'),
    'panel.fwd5_lag': ('可执行口径：t+1 收买入、持有 5 日收益', '小数', '信号 t 日收盘，成交 t+1 收盘'),
    'panel.y2': ('fwd5_lag 的截面去均值（建模用标签）', '小数', '同上'),
    'panel.y2_rk': ('y2 的当日截面分位', '分位', '同上'),
    'panel.industry': ('申万二级行业编码（-1=未分类）', '整数', '东财「所处行业」字段，按年铺开'),

    # ---- 42 因子（量价） ----
    'ret1': ('日收益率', '小数', 'hfq 收盘价；复权断层日置空'),
    'ret5': ('5 日动量', '小数', '窗口跨断层则置空'),
    'ret10': ('10 日动量', '小数', '同上'),
    'ret20': ('20 日动量', '小数', '同上'),
    'ret60': ('60 日动量', '小数', '同上'),
    'ret120': ('120 日动量', '小数', '同上'),
    'ret250': ('250 日动量（年度反转）', '小数', '同上；也用作「上市满 250 日」过滤器'),
    'std20': ('20 日收益率标准差（波动率）', '小数', '断层日 ret1 为空，rolling 自动跳过'),
    'std60': ('60 日收益率标准差', '小数', '同上'),
    'skew20': ('20 日收益率偏度', '—', '同上'),
    'kurt20': ('20 日收益率峰度', '—', '同上'),
    'maxret20': ('20 日单日最大涨幅（彩票效应）', '小数', '同上'),
    'down20': ('20 日下行波动率', '小数', '仅用负收益计算'),
    'turn5': ('5 日平均换手率', '%', '东财 turn 字段'),
    'turn20': ('20 日平均换手率', '%', '同上'),
    'turn60': ('60 日平均换手率', '%', '同上'),
    'turn_ratio': ('换手率短期/长期比（turn5 / turn60）', '倍数', '同上'),
    'turn_std20': ('20 日换手率标准差', '%', '同上'),
    'amp20': ('20 日平均振幅', '%', '东财 amplitude 字段'),
    'intra5': ('5 日平均日内收益（收盘/开盘−1）', '小数', '由 open/close 计算'),
    'intra20': ('20 日平均日内收益', '小数', '同上'),
    'overn5': ('5 日平均隔夜收益（开盘/昨收−1）', '小数', '同上'),
    'overn20': ('20 日平均隔夜收益', '小数', '同上'),
    'illiq20': ('20 日 Amihud 非流动性（|r|/成交额 ×1e8）', '1/元', 'Amihud (2002)'),
    'volratio': ('量比（5 日均量 / 60 日均量）', '倍数', '成交量（手）'),
    'pos250': ('250 日价格分位（(P−min)/(max−min)）', '0~1', '窗口跨断层则置空'),
    'dev20': ('相对 20 日均线偏离', '小数', '同上'),
    'dev60': ('相对 60 日均线偏离', '小数', '同上'),
    'size': ('规模代理（ln 20 日均成交额）', '对数元', '成交额（元）'),
    'momq60': ('风险调整动量（ret60 / std60）', '倍数', '同上'),
    'clv': ('收盘位置（(C−L)/(H−L)）', '0~1', '日内价格'),
    'ushadow': ('上影线占比（(H−max(O,C))/(H−L)）', '0~1', '同上'),
    'lshadow': ('下影线占比（(min(O,C)−L)/(H−L)）', '0~1', '同上'),
    'vwap_dev': ('收盘相对当日 VWAP 偏离', '小数', 'VWAP=成交额/(成交量×100)'),
    'pvcorr20': ('20 日价量相关性（ret1 vs 成交量变化）', '−1~1', '滚动 Pearson'),
    'pvcorr60': ('60 日价量相关性', '−1~1', '同上'),
    'amt_ratio': ('成交额比（20 日均 / 250 日均）', '倍数', '成交额（元）'),
    'turn_z': ('换手率 z-score（相对 60 日）', '—', '东财 turn'),
    'range_ratio': ('振幅收缩/扩张（amp20 / amp250）', '倍数', '东财 amplitude'),
    'mf20': ('20 日资金加权收益（ret1×amount 加权）', '小数', '成交额加权'),
    'wvma20': ('收盘相对 20 日成交额加权均价偏离', '小数', '窗口跨断层则置空'),
    'illiq_ratio': ('非流动性比（illiq20 / illiq250）', '倍数', 'Amihud'),

    # ---- 21 财务因子 ----
    'fin.roe_ttm': ('净资产收益率 TTM', '%', '财报，按公告日 PIT 对齐'),
    'fin.roa_ttm': ('总资产收益率 TTM', '%', '同上'),
    'fin.gpa': ('毛利/总资产（Novy-Marx 质量因子）', '小数', '同上'),
    'fin.gross_margin': ('毛利率 TTM', '%', '同上'),
    'fin.ocf_ni': ('经营现金流 / 净利润', '倍数', '盈利质量，>1 为好'),
    'fin.accrual': ('应计利润 / 总资产（Sloan 1996）', '小数', '低好'),
    'fin.debt_ratio': ('资产负债率', '%', '同上'),
    'fin.asset_turn': ('总资产周转率', '倍数', '同上'),
    'fin.roe_stab': ('过去 12 季 ROE 标准差', '—', '低好，衡量盈利稳定性'),
    'fin.roe_trend': ('ROE 趋势（近 4 季 vs 前 8 季）', '—', '同上'),
    'fin.gm_trend': ('毛利率趋势', '—', '同上'),
    'fin.rev_yoy': ('营收同比增速', '%', '同上'),
    'fin.profit_yoy': ('净利润同比增速', '%', '同上'),
    'fin.recv_gap': ('应收账款增速 − 营收增速（财务粉饰信号）', '%', '同上'),
    'fin.inv_gap': ('存货增速 − 营收增速', '%', '同上'),
    'fin.ocf_yoy': ('经营现金流同比增速', '%', '同上'),
    'fin.ep': ('盈利收益率（净利润 TTM / 总市值）', '小数', '总股本由「净利润÷EPS」反推'),
    'fin.bp': ('账面市值比（净资产 / 总市值）', '小数', '同上'),
    'fin.sp': ('营收市值比（营收 TTM / 总市值）', '小数', '同上'),
    'fin.ocf_yield': ('经营现金流收益率', '小数', '同上'),
    'fin.fcf_yield': ('自由现金流收益率（经营+投资现金流）/市值', '小数', '同上'),

    # ---- 其他数据集 ----
    'st.date': ('交易日', '—', 'baostock'),
    'st.symbol': ('股票代码', '—', 'baostock'),
    'st.is_st': ('是否 ST/*ST（1=是）', '0/1', 'baostock 时变字段'),
    'sw1.symbol': ('股票代码', '—', 'akshare 申万一级成分'),
    'sw1.l1': ('申万一级行业名称（31 个）', '—', '当前分类，无时变'),
    'sw2.code': ('股票代码', '—', '东财「所处行业」'),
    'sw2.sw_ind': ('申万二级行业名称（128 个）', '—', '同上'),
    'yjyg.symbol': ('股票代码', '—', '东财业绩预告'),
    'yjyg.date': ('公告日期', '—', '东财'),
    'yjyg.label': ('事件标签（业绩预喜/业绩预亏）', '—', '由预告类型映射'),
    'yjyg.note': ('预告类型 + 业绩变动原文', '—', '东财'),
    'ev.symbol': ('股票代码', '—', '事件表'),
    'ev.date': ('事件日', '—', '事件表'),
    'ev.label': ('事件类型（分组统计用）', '—', '事件表'),
    'ev.note': ('事件描述', '—', '事件表'),

    # yjyg 原始字段
    'yjyg.序号': ('行号（无意义）', '—', '东财接口'),
    'yjyg.股票代码': ('股票代码（6 位）', '—', '东财'),
    'yjyg.股票简称': ('股票名称', '—', '东财'),
    'yjyg.预测指标': ('预告的指标名（净利润/营收/每股收益等）', '—', '东财'),
    'yjyg.业绩变动': ('业绩变动原文描述', '—', '东财'),
    'yjyg.预测数值': ('预测数值原文', '—', '东财'),
    'yjyg.业绩变动幅度': ('业绩变动幅度（同比，%）', '%', '东财，用于排极端度'),
    'yjyg.业绩变动原因': ('公司给出的变动原因', '—', '东财'),
    'yjyg.预告类型': ('预增/略增/扭亏/续盈/预减/首亏/略减/续亏/不确定等', '—', '交易所强制披露口径'),
    'yjyg.上年同期值': ('上年同期数值原文', '—', '东财'),
    'yjyg.公告日期': ('公告日期', '—', '东财，事件日'),
    'yjyg.period': ('报告期（如 20240331）', '—', '文件名回填'),
    'st.source': ('数据来源标记', '—', 'baostock'),
    'sw2.year': ('数据年份', '—', '东财所处行业（当前分类，无时变）'),
}


def stats_series(s):
    """一个 Series 的统计量"""
    n = len(s)
    miss = float(s.isna().mean()) if n else np.nan
    v = ''
    if pd.api.types.is_datetime64_any_dtype(s):
        v = f'{s.min()} ~ {s.max()}（{s.nunique()} 个不同日期）'
    elif pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors='coerce')
        if x.notna().any():
            q = x.quantile([0.01, 0.5, 0.99])
            v = f'p1={q.iloc[0]:.4g} | 中位={q.iloc[1]:.4g} | p99={q.iloc[2]:.4g}'
    else:
        u = s.dropna().unique()
        v = f'{len(u)} 个取值，例：' + '、'.join(str(x)[:12] for x in u[:3])
    return n, miss, v


rows = []


def add(ds, key, df_or_arr, note=''):
    """ds=数据集名；key=META 前缀"""
    if isinstance(df_or_arr, np.ndarray):
        s = pd.Series(df_or_arr.ravel())
    else:
        s = df_or_arr
    n, miss, vr = stats_series(s)
    cn, unit, src = META.get(f'{key}.{getattr(df_or_arr, "name", None)}', ('', '', ''))
    rows.append(dict(数据集=ds, 字段=getattr(df_or_arr, 'name', ''), 中文含义=cn,
                     单位=unit, 类型=str(s.dtype), 缺失率=round(miss, 4),
                     取值范围=vr, 来源与时点=src, 备注=note))


# ---------------------------------------------------------------- 行情面板
print('扫描 panel_big…', flush=True)
pf = pd.read_parquet(f'{BIG}/panel_big.parquet')
FACTORS = ['ret1', 'ret5', 'ret10', 'ret20', 'ret60', 'ret120', 'ret250',
           'std20', 'std60', 'skew20', 'kurt20', 'maxret20', 'down20',
           'turn5', 'turn20', 'turn60', 'turn_ratio', 'turn_std20', 'amp20',
           'intra5', 'intra20', 'overn5', 'overn20', 'illiq20',
           'volratio', 'pos250', 'dev20', 'dev60', 'size', 'momq60',
           'clv', 'ushadow', 'lshadow', 'vwap_dev', 'pvcorr20', 'pvcorr60',
           'amt_ratio', 'turn_z', 'range_ratio', 'mf20', 'wvma20', 'illiq_ratio']
for c in ['date', 'symbol', 'y', 'y_rk', 'fwd5', 'fwd5_lag', 'y2', 'y2_rk', 'industry']:
    s = pf[c].copy(); s.name = c
    n, miss, vr = stats_series(s)
    cn, unit, src = META.get(f'panel.{c}', ('', '', ''))
    rows.append(dict(数据集='panel_big（行情面板）', 字段=c, 中文含义=cn, 单位=unit,
                     类型=str(s.dtype), 缺失率=round(miss, 4), 取值范围=vr,
                     来源与时点=src, 备注=f'{len(pf):,} 行'))
for c in FACTORS:
    s = pf[c].copy()
    n, miss, vr = stats_series(s)
    cn, unit, src = META.get(c, ('', '', ''))
    rows.append(dict(数据集='panel_big（行情面板）', 字段=c, 中文含义=cn, 单位=unit,
                     类型=str(s.dtype), 缺失率=round(miss, 4), 取值范围=vr,
                     来源与时点=src, 备注='截面 MAD 标准化后裁到 ±4'))

# ---------------------------------------------------------------- 财务面板
print('扫描 fin_panel…', flush=True)
fp = pd.read_parquet(f'{BIG}/fin_panel.parquet')
for c in fp.columns:
    if c in ('date', 'symbol'):
        cn = '交易日' if c == 'date' else '股票代码'
        s = fp[c]
        n, miss, vr = stats_series(s)
        rows.append(dict(数据集='fin_panel（财务面板）', 字段=c, 中文含义=cn, 单位='—',
                         类型=str(s.dtype), 缺失率=round(miss, 4), 取值范围=vr,
                         来源与时点='财报按公告日 PIT 对齐', 备注=f'{len(fp):,} 行'))
        continue
    s = fp[c]
    n, miss, vr = stats_series(s)
    cn, unit, src = META.get(f'fin.{c}', ('', '', '财报，PIT 对齐'))
    rows.append(dict(数据集='fin_panel（财务面板）', 字段=c, 中文含义=cn, 单位=unit,
                     类型=str(s.dtype), 缺失率=round(miss, 4), 取值范围=vr,
                     来源与时点=src, 备注='日频展开，数值跳变日=公告日'))

# ---------------------------------------------------------------- 其他
print('扫描其他…', flush=True)
for f, ds, key, src in [
    (f'{BIG}/st_calendar.parquet', 'st_calendar（ST 日历）', 'st', 'baostock 时变字段'),
    (f'{BIG}/sw_l1.parquet', 'sw_l1（申万一级）', 'sw1', 'akshare 申万成分'),
    (f'{BIG}/sw_industry.parquet', 'sw_industry（申万二级）', 'sw2', '东财所处行业'),
    (f'{BIG}/yjyg/yjyg_all.parquet', 'yjyg（业绩预告）', 'yjyg', '东财业绩预告'),
]:
    if not os.path.exists(f):
        continue
    d = pd.read_parquet(f)
    for c in d.columns:
        s = d[c]
        n, miss, vr = stats_series(s)
        cn, unit, src2 = META.get(f'{key}.{c}', ('', '', src))
        rows.append(dict(数据集=ds, 字段=c, 中文含义=cn, 单位=unit, 类型=str(s.dtype),
                         缺失率=round(miss, 4), 取值范围=vr, 来源与时点=src2,
                         备注=f'{len(d):,} 行'))

for f in sorted(glob.glob(f'{ROOT}/events/*.csv')):
    d = pd.read_csv(f, dtype=str)
    for c in d.columns:
        s = d[c]
        n, miss, vr = stats_series(s)
        cn, unit, src = META.get(f'ev.{c}', ('', '', '事件表'))
        rows.append(dict(数据集=f'events/{os.path.basename(f)}', 字段=c, 中文含义=cn,
                         单位=unit, 类型='str', 缺失率=round(miss, 4), 取值范围=vr,
                         来源与时点=src, 备注=f'{len(d)} 个事件'))

dd = pd.DataFrame(rows)
dd.to_csv(f'{ROOT}/reports/data_dictionary.csv', index=False, encoding='utf-8-sig')
print(f'已写 {len(dd)} 个字段 → reports/data_dictionary.csv')

# ---- markdown 版 ----
L = ['# 数据字典\n',
     f'> 自动生成于 {pd.Timestamp.now():%Y-%m-%d %H:%M}｜'
     f'共 {dd["数据集"].nunique()} 个数据集、{len(dd)} 个字段\n',
     '> 缺失率与取值范围为全量扫描结果；「来源与时点」说明该字段是否 PIT 安全。\n']
for ds, g in dd.groupby('数据集', sort=False):
    L.append(f'\n## {ds}\n')
    L.append('| 字段 | 中文含义 | 单位 | 类型 | 缺失率 | 取值范围 | 来源与时点 |')
    L.append('|---|---|---|---|---|---|---|')
    for _, r in g.iterrows():
        L.append(f"| `{r['字段']}` | {r['中文含义']} | {r['单位']} | {r['类型']} | "
                 f"{r['缺失率']*100:.1f}% | {r['取值范围']} | {r['来源与时点']} |")
open(f'{ROOT}/reports/data_dictionary.md', 'w', encoding='utf-8').write('\n'.join(L))
print('已写 reports/data_dictionary.md')
