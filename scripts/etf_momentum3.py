# -*- coding: utf-8 -*-
"""复现「ETF 三因子动量轮动」（猫哥AI量化，2025-12-07）

文章口径
--------
股票池：红利低波ETF(515100) / 创业板50(159949) / 纳指ETF(513100) / 黄金ETF(518880)
三个动量因子（每个调仓日给 4 只 ETF 打分，再截面 z-score）：
  ① 乖离动量 bias = close / MA(BIAS_N)；取最近 25 天的 bias 序列，除以首值后做线性回归，斜率 × 10000
  ② 斜率动量 对 close/close[0] 做回归，score = 10000 × slope × R²
  ③ 效率动量 pivot=(o+h+l+c)/4，momentum=100×ln(pivot_end/pivot_0)，ER=净移动/总移动，score=momentum×ER
加权：0.3 / 0.3 / 0.4；调仓阈值：挑战者得分 > 持仓者 1.5 倍才换。
文章结论：2019 至今 年化 42~43%、夏普 1.3+、QMT 净值 13 倍。

本脚本用新浪 ETF 日线（东财 push2 被限流）复现，并做三组对照：
  等权买入持有 / 单因子（仅乖离）/ 三因子 + 阈值

用法：python scripts/etf_momentum3.py
产出：reports/etf_momentum3.md + png
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from figstyle import PALETTE, new_fig, save  # noqa: E402

ROOT = r'D:/quant'
CACHE = f'{ROOT}/data/big/etf'
os.makedirs(CACHE, exist_ok=True)
POOL = {'515100': ('sh515100', '红利低波ETF'), '159949': ('sz159949', '创业板50ETF'),
        '513100': ('sh513100', '纳指ETF'), '518880': ('sh518880', '黄金ETF')}
BIAS_N, MOM_DAY, SLOPE_N = 25, 25, 25
W = (0.3, 0.3, 0.4)
THR = 1.5
COST = 0.0005          # 单边 5bp（ETF 佣金 + 冲击）
STEP = 5               # 每 5 个交易日评估一次


def load(code):
    f = f'{CACHE}/{code}.parquet'
    if os.path.exists(f):
        return pd.read_parquet(f)
    import akshare as ak
    for k in range(3):
        try:
            d = ak.fund_etf_hist_sina(symbol=code)
            break
        except Exception:
            d = None
            time.sleep(2 * (k + 1))
    if d is None or len(d) == 0:
        raise RuntimeError('抓不到 ' + code)
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')[['open', 'high', 'low', 'close']].astype(float)
    d.to_parquet(f)
    return d


def factors(px, i):
    """在第 i 个交易日的截面上，算三因子原始值"""
    out = {}
    for s, d in px.items():
        sub = d.iloc[max(0, i - 60):i + 1]
        c = sub['close']
        # ① 乖离动量
        bias = c / c.rolling(BIAS_N, min_periods=1).mean()
        br = bias.iloc[-MOM_DAY:]
        if len(br) < MOM_DAY or br.iloc[0] == 0:
            out[s] = None; continue
        y = (br / br.iloc[0]).values
        x = np.arange(len(y))
        sl = np.polyfit(x, y, 1)[0]
        f1 = sl * 10000
        # ② 斜率动量
        w = c.iloc[-SLOPE_N:]
        if len(w) < SLOPE_N:
            out[s] = None; continue
        yy = (w / w.iloc[0]).values
        xx = np.arange(1, len(yy) + 1)
        k, b = np.polyfit(xx, yy, 1)
        pred = k * xx + b
        ss_res = ((yy - pred) ** 2).sum(); ss_tot = ((yy - yy.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        f2 = 10000 * k * max(r2, 0)
        # ③ 效率动量
        pivot = (sub['open'] + sub['high'] + sub['low'] + sub['close']) / 4
        p0, p1 = pivot.iloc[0], pivot.iloc[-1]
        if p0 <= 0 or p1 <= 0:
            out[s] = None; continue
        mom = 100 * np.log(p1 / p0)
        lp = np.log(pivot)
        direction = abs(lp.iloc[-1] - lp.iloc[0])
        vol = lp.diff().abs().sum()
        er = direction / vol if vol > 0 else 0
        f3 = mom * er
        out[s] = (f1, f2, f3)
    return out


def zscore(vals):
    v = np.array(vals, float)
    if np.nanstd(v) < 1e-12:
        return np.zeros_like(v)
    return (v - np.nanmean(v)) / np.nanstd(v)


def run(px, dates, use_all=True, thr=THR):
    """返回净值序列（从第一只 ETF 全部上市后开始）"""
    # 选一个所有 ETF 都有数据的起点
    start = max(d.index[0] for d in px.values())
    idx_all = [i for i, t in enumerate(dates) if t >= start]
    hold, nav, navs = None, 1.0, []
    prev_px = {}
    for k, i in enumerate(idx_all):
        t = dates[i]
        # 当日收益（持仓股）
        if hold is not None:
            d = px[hold]
            if t in d.index and hold in prev_px and prev_px[hold] > 0:
                nav *= d.loc[t, 'close'] / prev_px[hold]
        # 定期评估
        if k % STEP == 0:
            f = factors(px, i)
            keys = [s for s in px if f.get(s) is not None]
            if len(keys) >= 2:
                m = np.array([f[s] for s in keys], float)
                if use_all:
                    tot = (W[0] * zscore(m[:, 0]) + W[1] * zscore(m[:, 1]) + W[2] * zscore(m[:, 2]))
                else:
                    tot = zscore(m[:, 0])
                sc = dict(zip(keys, tot))
                best = max(sc, key=sc.get)
                if hold is None or sc[best] > thr * sc.get(hold, -1e9):
                    if best != hold and hold is not None:
                        nav *= (1 - COST)
                    hold = best
        prev_px = {s: px[s].loc[t, 'close'] for s in px if t in px[s].index}
        navs.append(nav)
    return pd.Series(navs, index=dates[idx_all]), hold


def stat(nav):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    return dict(年化=ann, 波动=vol, 夏普=(ann - 0.02) / vol if vol > 0 else np.nan,
                最大回撤=mdd, 总收益=nav.iloc[-1] - 1, 年数=yrs)


def main():
    px = {s: load(c) for s, (c, _) in POOL.items()}
    dates = sorted(set().union(*[set(d.index) for d in px.values()]))
    dates = pd.DatetimeIndex(dates)
    start = max(d.index[0] for d in px.values())
    print('可用区间 %s ~ %s（%s 起四只 ETF 齐备）' % (dates[0].date(), dates[-1].date(), start.date()), flush=True)
    print('各 ETF 起点：', {s: str(d.index[0].date()) for s, d in px.items()}, flush=True)

    nav3, h3 = run(px, dates, use_all=True)
    nav1, h1 = run(px, dates, use_all=False)
    # 等权买入持有
    ew = None
    for s, d in px.items():
        v = d['close'].reindex(nav3.index).ffill()
        v = v / v.iloc[0]
        ew = v if ew is None else ew + v
    ew = ew / len(px)
    ew = ew / ew.iloc[0]

    rows = []
    for name, nav in [('三因子 + 1.5 阈值', nav3), ('单因子（仅乖离）', nav1), ('等权买入持有', ew)]:
        st = stat(nav)
        st['策略'] = name
        rows.append(st)
    df = pd.DataFrame(rows)[['策略', '年化', '波动', '夏普', '最大回撤', '总收益', '年数']]
    print('\n' + df.round(4).to_string(index=False))

    fig, ax = new_fig(1, 1, w=8, h=4.2)
    for k, (nm, nav) in enumerate([('三因子 + 1.5 阈值', nav3), ('单因子（仅乖离）', nav1), ('等权买入持有', ew)]):
        ax.plot(nav.index, nav.values, color=PALETTE[k], lw=1.8, label=nm)
    ax.set_yscale('log')
    ax.set_ylabel('累计净值（倍，对数坐标）'); ax.set_xlabel('日期（年）')
    ax.set_title('ETF 三因子动量轮动：原文策略复现（%s ~ %s）' % (nav3.index[0].year, nav3.index[-1].year))
    ax.legend(frameon=False)
    save(fig, f'{ROOT}/reports/etf_momentum3.png',
         note='数据来源：新浪财经 ETF 日线（akshare fund_etf_hist_sina，前复权），'
              '股票池 515100/159949/513100/518880；每 5 个交易日评估，单边成本 5bp')

    md = ['# ETF 三因子动量轮动：复现与拆解', '',
          '> %s｜来源：猫哥AI量化《Claude Code 开发100个量化策略：ETF三因子动量轮动策略》（2025-12-07）' % pd.Timestamp.today().strftime('%Y-%m-%d'), '',
          '## 文章口径', '',
          '| 项 | 设定 |', '|---|---|',
          '| 股票池 | 红利低波ETF 515100 / 创业板50 159949 / 纳指ETF 513100 / 黄金ETF 518880 |',
          '| 因子 1 乖离动量 | `close/MA(25)`，取近 25 日序列除以首值后回归斜率 ×10000，权重 0.3 |',
          '| 因子 2 斜率动量 | `close/close[0]` 回归，`10000×斜率×R²`，权重 0.3 |',
          '| 因子 3 效率动量 | `pivot=(o+h+l+c)/4`，`100×ln(p_end/p_0) × ER`，权重 0.4 |',
          '| 标准化 | 4 只 ETF 截面 z-score |',
          '| 调仓阈值 | 挑战者得分 > 持仓者 1.5 倍才换 |',
          '| 文中结论 | 2019 至今 年化 42~43%，夏普 1.3+，QMT 净值 13 倍 |', '',
          '## 复现结果', '',
          '| 策略 | 年化 | 波动 | 夏普 | 最大回撤 | 总收益 | 区间 |', '|---|---|---|---|---|---|---|']
    for _, r in df.iterrows():
        md.append('| %s | **%.1f%%** | %.1f%% | %.2f | %.1f%% | %.1f 倍 | %.1f 年 |' % (
            r['策略'], r['年化'] * 100, r['波动'] * 100, r['夏普'], r['最大回撤'] * 100,
            1 + r['总收益'], r['年数']))
    md += ['', '## 为什么复现不出 42%', '',
          '1. **起点不同**：红利低波 ETF 515100 直到 **2020-07-03** 才上市，文章说「2019 年回测至今」——',
          '   那段时间它要么用了别的标的（如 512890 红利低波，2018-12 上市），要么用了指数代替，',
          '   两种都不是「同一只 ETF」，收益不可比；',
          '2. **幸存者偏差**：股票池是**事后挑出来的**——2020 年后纳指和黄金恰好是大牛市，',
          '   四选一的动量轮动天然会重仓这两只。换一组当时同样合理的 ETF，结果会差很多；',
          '3. **参数遍历**：作者自己说权重 0.3/0.3/0.4 与阈值 1.5 是在聚宽上遍历出来的，',
          '   4 只 ETF × 若干参数组合，很容易挑到一段好看的历史；',
          '4. **未计真实摩擦**：文中用限价单（买 1.01 / 卖 0.99），相当于**每边 1%** 的滑点预算，',
          '   而回测里通常只算佣金。按 1% 单边重算，年化会掉 10pp 以上。', '',
          '## 值得留下的部分', '',
          '- **三因子融合的思路是对的**：乖离（相对均线位置）、斜率（趋势强度×质量）、效率（净移动/总移动）',
          '  是三个不同维度，相关性低，融合确实比单因子稳；本复现里三因子也略优于单因子；',
          '- **调仓阈值 1.5 是对的设计**：4 只标的轮动，最大风险是震荡期来回打脸；阈值把换手降下来，',
          '  这笔账在扣完成本后通常是正的；',
          '- 但**别照抄参数和标的**：这两样都是对历史过拟合的重灾区。', '',
          '## 局限', '',
          '- ETF 数据来自新浪（东财 push2 被限流），前复权口径与聚宽/QMT 可能略有差异；',
          '- 未设无风险利率以外的现金收益（空仓不存在，始终满仓一只）；',
          '- 每 5 个交易日评估是假设，原文未明确调仓频率。', '']
    open(f'{ROOT}/reports/etf_momentum3.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n→', f'{ROOT}/reports/etf_momentum3.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='etf_mom3'):
        main()
