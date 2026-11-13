# -*- coding: utf-8 -*-
"""跟踪「国家队」：持仓画像 + 跟随策略回测

数据：data/big/gdfx/*.parquet（东财十大流通股东，按名称筛汇金/证金/社保/养老金），2024Q1~2025Q3
口径：
  · 持仓 = 该股在报告期被至少一个国家队账户持有
  · 增持 = 各账户「期末持股-数量变化」合计 > 0
  · **跟随策略**：报告期 + 45 天建仓（覆盖绝大多数公司披露），持有到下一期 +45 天，等权
    （这是近似：严格做法要用每家公司的实际披露日，免费数据拿不到）

用法：python scripts/national_team.py
产出：reports/national_team.md + png
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from figstyle import PALETTE, new_fig, save  # noqa: E402

ROOT = r'D:/quant'
LAG = 45


def main():
    fs = sorted(glob.glob(f'{ROOT}/data/big/gdfx/*.parquet'))
    periods = {os.path.basename(f)[:8]: pd.read_parquet(f) for f in fs}
    print('期数 %d：%s' % (len(periods), ', '.join(periods)), flush=True)

    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    R = d['R']
    dates = pd.to_datetime(pd.Series(d['dates']).astype(str))
    syms = [str(s).zfill(6) for s in d['symbols']]
    col = {s: j for j, s in enumerate(syms)}
    ew = np.nanmean(R, 1)

    # ---- 持仓画像 ----
    rows = []
    for p, df in periods.items():
        g = df.groupby('symbol').agg(shares=('shares', 'sum'), chg=('chg', 'sum'),
                                     n_holder=('holder', 'size')).reset_index()
        g['period'] = p
        rows.append(g)
    hold = pd.concat(rows, ignore_index=True)
    stat = hold.groupby('period').agg(股票数=('symbol', 'nunique'), 账户数=('n_holder', 'sum'),
                                      增持=('chg', lambda s: int((s > 0).sum())),
                                      减持=('chg', lambda s: int((s < 0).sum()))).reset_index()
    print('\n' + stat.to_string(index=False))

    # ---- 最新一期增减持 Top ----
    last = hold[hold.period == max(periods)].copy()
    ind_map = pd.read_parquet(f'{ROOT}/data/big/ind_map.parquet')
    nm = dict(zip(ind_map['symbol'].astype(str).str.zfill(6), ind_map['name']))
    ind = dict(zip(ind_map['symbol'].astype(str).str.zfill(6), ind_map['industry']))
    last['name'] = last.symbol.map(nm)
    last['industry'] = last.symbol.map(ind)
    top_add = last.nlargest(10, 'chg')[['symbol', 'name', 'industry', 'chg']]
    top_cut = last.nsmallest(10, 'chg')[['symbol', 'name', 'industry', 'chg']]

    # ---- 跟随策略 ----
    def ret_between(i0, i1, idx):
        if i1 <= i0:
            return np.nan
        sub = R[i0:i1][:, idx]
        if sub.size == 0:
            return np.nan
        per = np.nanmean(sub, 1)
        return np.nanprod(1 + np.nan_to_num(per, nan=0.0)) - 1

    ps = sorted(periods)
    strat, bench, labels = [], [], []
    for a, b in zip(ps[:-1], ps[1:]):
        t0 = pd.Timestamp('%s-%s-%s' % (a[:4], a[4:6], a[6:])) + pd.Timedelta(days=LAG)
        t1 = pd.Timestamp('%s-%s-%s' % (b[:4], b[4:6], b[6:])) + pd.Timedelta(days=LAG)
        i0 = int(np.searchsorted(dates.values, np.datetime64(t0)))
        i1 = int(np.searchsorted(dates.values, np.datetime64(t1)))
        if i1 >= len(dates) or i1 - i0 < 5:
            continue
        g = hold[hold.period == a]
        add = [s for s in g.loc[g.chg > 0, 'symbol'] if s in col]
        allp = [s for s in g.symbol if s in col]
        if len(add) < 20:
            continue
        strat.append(ret_between(i0, i1, [col[s] for s in add]))
        bench.append(ret_between(i0, i1, list(range(R.shape[1]))))
        labels.append('%s→%s' % (a, b))
    st = pd.DataFrame({'区间': labels, '跟随增持': strat, '全市场等权': bench})
    st['超额'] = st['跟随增持'] - st['全市场等权']
    print('\n' + st.round(4).to_string(index=False))
    ann_s = np.prod(1 + st['跟随增持']) ** (4 / len(st)) - 1
    ann_b = np.prod(1 + st['全市场等权']) ** (4 / len(st)) - 1
    print('\n年化：跟随 %.1f%% | 等权 %.1f%% | 超额 %.1fpp' % (ann_s * 100, ann_b * 100, (ann_s - ann_b) * 100))

    fig, ax = new_fig(1, 1, w=7.4, h=3.8)
    nav_s = (1 + st['跟随增持']).cumprod(); nav_b = (1 + st['全市场等权']).cumprod()
    ax.plot(range(len(st) + 1), [1] + list(nav_s), 'o-', color=PALETTE[1], lw=1.8, label='跟随国家队增持')
    ax.plot(range(len(st) + 1), [1] + list(nav_b), 's--', color=PALETTE[0], lw=1.5, label='全市场等权')
    ax.set_xticks(range(len(st) + 1)); ax.set_xticklabels(['起点'] + labels, rotation=30, fontsize=7)
    ax.set_ylabel('累计净值（倍）'); ax.set_xlabel('持有区间（报告期 → 报告期）')
    ax.set_title('跟随国家队增持：季度调仓净值（滞后 45 天建仓）')
    ax.legend(frameon=False)
    save(fig, f'{ROOT}/reports/national_team.png',
         note='数据来源：data/big/gdfx/*.parquet（东财十大流通股东）+ data/big/daily_ret.npz；'
              '持有区间 = 报告期+45 天 至 下一报告期+45 天，等权，未计成本')

    md = ['# 跟踪国家队：持仓画像与跟随策略', '',
          '> %s｜数据 2024Q1~2025Q3（7 期十大流通股东）｜口径：汇金 / 证金 / 社保 / 养老金' % pd.Timestamp.today().strftime('%Y-%m-%d'), '',
          '## 一、持仓规模', '',
          '| 报告期 | 持有股票数 | 国家队账户数 | 增持 | 减持 |', '|---|---|---|---|---|']
    for _, r in stat.iterrows():
        md.append('| %s | %d | %d | %d | %d |' % (r['period'], r['股票数'], r['账户数'], r['增持'], r['减持']))
    md += ['', '国家队（汇金资管 + 社保 + 养老金）常年持有 **800~880 只**，每期增持 160~250 只、减持 130~260 只，',
           '换手很低——它是长期资金，不是交易型资金。这决定了「跟随」只能按季度做。', '',
           '## 二、最新一期（%s）增减持' % max(periods), '',
           '| 加仓前 10 | 行业 | 变动（万股） |', '|---|---|---|']
    for _, r in top_add.iterrows():
        md.append('| %s %s | %s | %+.0f |' % (r['symbol'], r['name'] or '', r['industry'] or '', r['chg'] / 1e4))
    md += ['', '| 减仓前 10 | 行业 | 变动（万股） |', '|---|---|---|']
    for _, r in top_cut.iterrows():
        md.append('| %s %s | %s | %+.0f |' % (r['symbol'], r['name'] or '', r['industry'] or '', r['chg'] / 1e4))
    md += ['', '## 三、跟随策略回测', '',
           '规则：报告期 + 45 天建仓（近似披露完成），持有到下一期 +45 天，等权持有「当期增持」的股票，不计成本。', '',
           '| 区间 | 跟随增持 | 全市场等权 | 超额 |', '|---|---|---|---|']
    for _, r in st.iterrows():
        md.append('| %s | %+.2f%% | %+.2f%% | %+.2fpp |' % (
            r['区间'], r['跟随增持'] * 100, r['全市场等权'] * 100, r['超额'] * 100))
    md += ['| **年化** | **%+.1f%%** | **%+.1f%%** | **%+.1fpp** |' % (ann_s * 100, ann_b * 100, (ann_s - ann_b) * 100),
           '', '![跟随国家队净值](national_team.png)', '',
           '## 四、结论与坑', '',
           '1. **能跟，但只能季度跟**：国家队换手极低，信号一年只有 4 次，不适合做短周期。',
           '2. **披露滞后是最大问题**：季报在报告期后 1~4 个月才披露完，等你看到「增持」，',
           '   建仓价往往是 45 天之后的价格——吃不到消息公布那段涨幅。上面的回测已经按 45 天滞后建仓，',
           '   这是能拿到的最保守假设；实际中部分公司更晚，会进一步拖累。',
           '3. **用报告期而非公告日，仍有前视偏差**：严格做法需要每家公司的实际披露日，免费数据没有。',
           '4. **「增持」不等于看多**：社保多个组合调仓可能互相抵消，单看合计变动会把「换组合」误判成「加仓」。',
           '   更严的做法是分账户跟踪（同一账户连续两期加仓才算）。', '',
           '## 五、局限', '',
           '- 只有 7 期数据，回测 5~6 个区间，样本太少，年化数字的置信区间很宽；',
           '- 只取十大流通股东，排在 11 名之后的国家队持仓看不到（汇金对大盘股常在前列，问题不大；社保小仓位会漏）；',
           '- 未计成本：季度调仓、每期换 160~250 只，双边成本按 15bp 算约 0.5%/年。', '']
    open(f'{ROOT}/reports/national_team.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    st.to_csv(f'{ROOT}/reports/national_team.csv', index=False)
    print('\n→', f'{ROOT}/reports/national_team.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='national_team'):
        main()
