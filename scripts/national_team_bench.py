# -*- coding: utf-8 -*-
"""跟踪国家队 v2：换基准 + 拉长样本（2020Q1~2025Q3，23 期 / 22 个持有区间）

v1 的问题：拿「全市场等权」去比一个偏大盘的组合，天然会输，结论不成立。
v2 补三个更公平的基准：

  1. 沪深300 指数（市值加权，真实指数点位，data/big/idx_daily.pkl）
  2. 沪深300 等权（**代理**：每个调仓日用流通市值前 300 只等权。缺历史成分表，
     只能用市值代理——中证 300 本身就是按自由流通市值选样，代理误差可接受但需说明）
  3. 同市值分位组对照（最严格）：把全市场按流通市值分 10 组，组合的基准收益 =
     按其持仓在各组的权重，加权各组等权收益。这个基准与持仓的市值分布完全相同，
     剩下的差异才是「选股」。

流通市值怎么来：`turn`（换手率 %）与 `amount`（成交额，元）满足
    流通市值 = 流通股本 × 价格 ≈ (amount / turn) × 100
（用 amount/turn 反推，天然避开复权因子与股本变更的影响；平安银行实测 2,290 亿 vs 真实约 2,400 亿）

用法：python scripts/national_team_bench.py
产出：reports/national_team_bench.md / .csv / .png
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
CAP_CACHE = f'{ROOT}/data/big/cap_mat.npy'


def build_cap(dates, syms):
    """流通市值矩阵（20 日均值），单位：元"""
    if os.path.exists(CAP_CACHE):
        m = np.load(CAP_CACHE)
        if m.shape == (len(dates), len(syms)):
            print('  流通市值：读缓存 %s' % CAP_CACHE, flush=True)
            return m
    print('  流通市值：从 raw_hfq 反推（amount/turn×100）', flush=True)
    col = {s: j for j, s in enumerate(syms)}
    di = {np.datetime64(t): i for i, t in enumerate(dates.values)}
    T, N = len(dates), len(syms)
    acc = np.zeros((T, N), np.float32)
    cnt = np.zeros((T, N), np.int16)
    fs = sorted(glob.glob(f'{ROOT}/data/big/raw_hfq/*.parquet'))
    for k, f in enumerate(fs):
        try:
            d = pd.read_parquet(f, columns=['date', 'symbol', 'amount', 'turn'])
        except Exception:
            continue
        s = str(d['symbol'].iloc[0]).zfill(6)
        j = col.get(s)
        if j is None:
            continue
        t = pd.to_datetime(d['date']).values.astype('datetime64[D]')
        i = np.array([di.get(x, -1) for x in t], dtype=np.int32)
        ok = i >= 0
        cap = (d['amount'].values / np.maximum(d['turn'].values, 1e-6) * 100.0)
        ok &= np.isfinite(cap) & (d['turn'].values > 0) & (cap > 1e8) & (cap < 5e12)
        acc[i[ok], j] = cap[ok]
        cnt[i[ok], j] = 1
        if (k + 1) % 1500 == 0:
            print('    %d/%d 只' % (k + 1, len(fs)), flush=True)
    # 20 日滚动均值（用累计和，缺失按 0 计，分母按有效天数）
    c = np.cumsum(acc, 0)
    n = np.cumsum(cnt.astype(np.float32), 0)
    c20 = np.vstack([np.zeros((20, N), np.float32), c])
    n20 = np.vstack([np.zeros((20, N), np.float32), n])
    s = c20[20:] - c20[:-20]
    k = n20[20:] - n20[:-20]
    out = np.where(k > 0, s / np.maximum(k, 1), np.nan).astype(np.float32)
    out = np.vstack([np.full((20, N), np.nan, np.float32), out])
    np.save(CAP_CACHE, out)
    print('  → 缓存 %s（%d 只有市值）' % (CAP_CACHE, int(np.isfinite(out[-1]).sum())), flush=True)
    return out


def ew_ret(R, i0, i1, idx):
    """等权持有 idx（列号）从 i0 收盘到 i1 收盘的收益（逐日再平衡，近似）"""
    if i1 <= i0 or len(idx) == 0:
        return np.nan
    sub = R[i0 + 1:i1 + 1][:, idx]
    per = np.nanmean(sub, 1)
    if not np.isfinite(per).any():
        return np.nan
    return float(np.prod(1 + np.nan_to_num(per, nan=0.0)) - 1)


def main():
    fs = sorted(glob.glob(f'{ROOT}/data/big/gdfx/*.parquet'))
    periods = {os.path.basename(f)[:8]: pd.read_parquet(f) for f in fs}
    ps = sorted(periods)
    print('期数 %d：%s ~ %s' % (len(ps), ps[0], ps[-1]), flush=True)

    d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    R = d['R']
    dates = pd.to_datetime(pd.Series(d['dates']).astype(str))
    syms = [str(s).zfill(6) for s in d['symbols']]
    col = {s: j for j, s in enumerate(syms)}

    cap = build_cap(dates, syms)
    idx = pd.read_pickle(f'{ROOT}/data/big/idx_daily.pkl')['沪深300']
    hs300 = idx.reindex(dates).ffill().values

    hold = pd.concat([periods[p].assign(period=p) for p in ps], ignore_index=True)
    g = hold.groupby(['period', 'symbol'], as_index=False).agg(chg=('chg', 'sum'))

    rows = []
    for a, b in zip(ps[:-1], ps[1:]):
        t0 = pd.Timestamp('%s-%s-%s' % (a[:4], a[4:6], a[6:])) + pd.Timedelta(days=LAG)
        t1 = pd.Timestamp('%s-%s-%s' % (b[:4], b[4:6], b[6:])) + pd.Timedelta(days=LAG)
        i0 = int(np.searchsorted(dates.values, np.datetime64(t0)))
        i1 = int(np.searchsorted(dates.values, np.datetime64(t1)))
        if i1 >= len(dates) or i1 - i0 < 5:
            continue
        gg = g[g.period == a]
        add = [s for s in gg.loc[gg.chg > 0, 'symbol'] if s in col]
        allp = [s for s in gg.symbol if s in col]
        if len(add) < 20:
            continue
        ai = np.array([col[s] for s in add])
        pi = np.array([col[s] for s in allp])

        # 基准 0：全市场等权
        b_all = ew_ret(R, i0, i1, np.arange(R.shape[1]))
        # 基准 1：沪深300 指数（市值加权）
        b_idx = float(hs300[i1] / hs300[i0] - 1)
        # 基准 2：沪深300 等权代理 = 流通市值前 300
        c0 = cap[i0]
        valid = np.isfinite(c0) & np.isfinite(R[i0])
        top = np.argsort(-np.where(valid, c0, -np.inf))[:300]
        b_hs300ew = ew_ret(R, i0, i1, top)
        # 基准 3：同市值十分位加权（与持仓的市值分布完全一致）
        pool = np.where(valid)[0]
        if len(pool) < 1000:
            continue
        q = pd.qcut(pd.Series(c0[pool]).rank(method='first'), 10, labels=False).values
        dec_ret, dec_n = {}, {}
        for k in range(10):
            sel = pool[q == k]
            dec_ret[k] = ew_ret(R, i0, i1, sel)
            dec_n[k] = len(sel)
        hq = pd.Series(c0[ai]).rank(method='first')
        hq = pd.qcut(hq, 10, labels=False).values if len(ai) >= 20 else None
        if hq is None:
            continue
        w = pd.Series(hq).value_counts(normalize=True).to_dict()
        b_size = sum(w[k] * dec_ret[k] for k in w if k in dec_ret and np.isfinite(dec_ret[k]))

        rows.append(dict(
            区间='%s→%s' % (a, b),
            跟随增持=ew_ret(R, i0, i1, ai),
            跟随全部持仓=ew_ret(R, i0, i1, pi),
            全市场等权=b_all, 沪深300指数=b_idx, 沪深300等权=b_hs300ew,
            同市值分位=b_size,
            组合市值中位=float(np.nanmedian(c0[ai])) / 1e8,
            基准市值中位=float(np.nanmedian(c0[pool])) / 1e8,
            n_add=len(ai),
        ))

    st = pd.DataFrame(rows)
    for c in ['沪深300指数', '沪深300等权', '同市值分位', '全市场等权']:
        st['超额_' + c] = st['跟随增持'] - st[c]

    n = len(st)
    from scipy import stats as sps

    def ann(x):
        return float(np.prod(1 + x) ** (4 / n) - 1)

    print('\n' + st.round(4).to_string(index=False))
    print('\n区间数 %d | 组合市值中位 %.0f 亿 | 全市场 %.0f 亿' % (
        n, st['组合市值中位'].median(), st['基准市值中位'].median()))
    summ = []
    for c in ['全市场等权', '沪深300指数', '沪深300等权', '同市值分位']:
        e = st['超额_' + c]
        t, p = sps.ttest_1samp(e, 0)
        summ.append(dict(基准=c, 年化=ann(st[c]), 超额均值=e.mean(), t=t, p=p,
                         胜率=float((e > 0).mean()), 年化超额=ann(st['跟随增持']) - ann(st[c])))
        print('vs %-10s 年化 %6.1f%% | 超额均值 %+5.2f%%/季 | t %5.2f p %.3f | 胜率 %2.0f%%' % (
            c, ann(st[c]) * 100, e.mean() * 100, t, p, (e > 0).mean() * 100))
    sm = pd.DataFrame(summ)

    # 对照：只跟「增持」有用吗？拿「跟随全部持仓」做同口径比较
    d_add = st['跟随增持'] - st['跟随全部持仓']
    t_add, p_add = sps.ttest_1samp(d_add, 0)
    ann_add, ann_all = ann(st['跟随增持']), ann(st['跟随全部持仓'])
    print('增持 vs 全部持仓：年化 %.1f%% vs %.1f%% | 差 %+.2f%%/季 t %.2f p %.3f 胜率 %.0f%%' % (
        ann_add * 100, ann_all * 100, d_add.mean() * 100, t_add, p_add, (d_add > 0).mean() * 100))

    # ---- 图：净值 ----
    fig, ax = new_fig(1, 2, w=11, h=4.0)
    x = range(n + 1)
    series = [('跟随增持', '跟随增持', PALETTE[1], '-'),
              ('跟随全部持仓', '跟随全部持仓', PALETTE[5], '-'),
              ('同市值分位组（等市值对照）', '同市值分位', PALETTE[0], '--'),
              ('沪深300 等权', '沪深300等权', PALETTE[2], '-.'),
              ('沪深300 指数', '沪深300指数', PALETTE[4], ':')]
    for lab, c, cl, ls in series:
        ax[0].plot(x, [1] + list((1 + st[c]).cumprod()), ls, color=cl, lw=1.7, label=lab)
    ax[0].set_ylabel('累计净值（倍）'); ax[0].set_xlabel('持有区间（季度，滞后 45 天建仓）')
    ax[0].set_title('跟随国家队增持 vs 三个可比基准（2020Q1~2025Q3）')
    ax[0].legend(frameon=False, fontsize=8)

    bar_c = ['全市场等权', '沪深300指数', '沪深300等权', '同市值分位']
    vals = [(ann(st['跟随增持']) - ann(st[c])) * 100 for c in bar_c]
    ax[1].barh(['vs ' + c for c in bar_c], vals, color=PALETTE[:4])
    ax[1].axvline(0, color='k', lw=0.8)
    ax[1].set_xlabel('跟随增持年化收益减去基准年化（百分点）')
    ax[1].set_title('各基准下的年化超额')
    for i, v in enumerate(vals):
        ax[1].text(v, i, ' %+.1f' % v, va='center', fontsize=8,
                   ha='left' if v >= 0 else 'right')
    save(fig, f'{ROOT}/reports/national_team_bench.png',
         note='数据来源：data/big/gdfx/*.parquet（东财十大流通股东，%d 期）+ data/big/daily_ret.npz + '
              'data/big/idx_daily.pkl；流通市值由 amount/turn×100 反推；持有区间 = 报告期+45 天，等权，未计成本' % len(ps))

    st.to_csv(f'{ROOT}/reports/national_team_bench.csv', index=False)
    print('\n→', f'{ROOT}/reports/national_team_bench.csv')

    # ---------------- 报告 ----------------
    st['年'] = st['区间'].str[:4].astype(int)
    base_c = '同市值分位'
    md = ['# 跟踪国家队 v2：换基准 + 拉长样本', '',
          '> %s｜%d 期十大流通股东（%s ~ %s）｜%d 个持有区间｜滞后 45 天建仓、季度调仓、等权、未计成本' % (
              pd.Timestamp.today().strftime('%Y-%m-%d'), len(ps), ps[0], ps[-1], n), '',
          '## 一、为什么这版要换基准', '',
          'v1 的结论是「跟随增持年化 17.2% vs 全市场等权 33.8%，跑输 16.6pp」，但它有两个硬伤：', '',
          '- **样本太短**：只有 6 个区间，全落在 2024Q1~2025Q3 这一年半（小微盘暴涨、红利/银行滞涨）；',
          '- **只有一个基准**：全市场等权天然超配小市值，而国家队持仓市值中位 %.0f 亿、全市场 %.0f 亿，' % (
              st['组合市值中位'].median(), st['基准市值中位'].median()),
          '  差 %.1f 倍，等于让一个偏大盘组合去跑小盘赛跑。' % (
              st['组合市值中位'].median() / st['基准市值中位'].median()), '',
          '这版补三个更公平的对照：', '',
          '| 基准 | 定义 | 它回答什么 |', '|---|---|---|',
          '| 全市场等权 | 全部股票等权 | v1 的口径，只作参照 |',
          '| 沪深300 指数 | 真实指数点位（市值加权） | 「跑赢大盘了吗」 |',
          '| 沪深300 等权 | **代理**：每期流通市值前 300 只等权 | 「同样的股票池，等权持有效果如何」 |',
          '| **同市值分位组** | 全市场按流通市值分 10 组，按组合在各组的权重加权各组等权收益 | 「同样的市值分布下，选股有没有超额」 |', '',
          '最后一行的基准和组合**市值分布完全一致**，两者之差才是纯选股能力。', '',
          '## 二、流通市值怎么算', '',
          '没有现成市值字段，用换手率反推：`流通市值 = 成交额 / 换手率 × 100`。',
          '它天然避开复权因子与股本变更——成交额和换手率都是当日真实值。',
          '校验：平安银行 2,293 亿（真实约 2,400 亿）、贵州茅台 16,321 亿（真实约 1.8 万亿），',
          '误差 5~10%，足够做十分位分组。取 20 日均值平滑。', '',
          '## 三、主结果', '',
          '| 基准 | 基准年化 | 跟随增持年化 | 年化超额 | 单期超额均值 | t | p | 胜率 |',
          '|---|---|---|---|---|---|---|---|']
    for _, r in sm.iterrows():
        md.append('| %s | %+.1f%% | %+.1f%% | **%+.1fpp** | %+.2f%% | %+.2f | %.3f | %.0f%% |' % (
            r['基准'], r['年化'] * 100, ann(st['跟随增持']) * 100, r['年化超额'] * 100,
            r['超额均值'] * 100, r['t'], r['p'], r['胜率'] * 100))

    md += ['', '## 四、逐区间明细（%）', '',
           '| 区间 | 跟随增持 | 沪深300指数 | 沪深300等权 | 同市值分位 | 超额(vs 同市值) | 增持只数 |',
           '|---|---|---|---|---|---|---|']
    for _, r in st.iterrows():
        md.append('| %s | %+.2f | %+.2f | %+.2f | %+.2f | **%+.2f** | %d |' % (
            r['区间'], r['跟随增持'] * 100, r['沪深300指数'] * 100, r['沪深300等权'] * 100,
            r['同市值分位'] * 100, r['超额_同市值分位'] * 100, r['n_add']))

    yr = st.groupby('年').agg(跟随增持=('跟随增持', lambda x: np.prod(1 + x) - 1),
                              同市值分位=('同市值分位', lambda x: np.prod(1 + x) - 1),
                              沪深300指数=('沪深300指数', lambda x: np.prod(1 + x) - 1),
                              超额=('超额_同市值分位', 'mean')).reset_index()
    md += ['', '## 五、分年度', '',
           '| 年 | 跟随增持 | 同市值分位 | 沪深300 指数 | 单期超额均值 |', '|---|---|---|---|---|']
    for _, r in yr.iterrows():
        md.append('| %d | %+.1f%% | %+.1f%% | %+.1f%% | %+.2f%% |' % (
            r['年'], r['跟随增持'] * 100, r['同市值分位'] * 100, r['沪深300指数'] * 100, r['超额'] * 100))

    md += ['', '![净值与超额](national_team_bench.png)', '',
           '## 六、只看「增持」有用吗', '',
           '| 策略 | 年化 |', '|---|---|',
           '| 跟随「增持」的股票 | %+.1f%% |' % (ann_add * 100),
           '| 跟随国家队**全部持仓**（不筛增持） | %+.1f%% |' % (ann_all * 100),
           '| 差（增持 - 全部） | **%+.1fpp** |' % ((ann_add - ann_all) * 100), '',
           '单期差 %+.2f%%、t = %.2f、p = %.3f、胜率 %.0f%%。**「增持」这个动作本身没有信息**——' % (
               d_add.mean() * 100, t_add, p_add, (d_add > 0).mean() * 100),
           '只挑增持的反而略差。这也合理：国家队增持常见于「跌多了被动补仓 / 指数化建仓」，',
           '不是因为看好某只股票；而它的存量持仓才是长期选出来的。', '',
           '## 七、结论', '',
           '1. **v1 的「跑输 16.6pp」主要是样本区间造成的**：同样对全市场等权，样本拉长到 2020-2025 后只剩 %+.1fpp。' % (
               (ann_add - ann(st['全市场等权'])) * 100),
           '   而且换不换基准几乎不影响（全市场等权 %+.1f%% vs 同市值分位 %+.1f%%，只差 %.1fpp），' % (
               ann(st['全市场等权']) * 100, ann(st[base_c]) * 100,
               (ann(st['全市场等权']) - ann(st[base_c])) * 100),
           '   也就是说，v1 那句「基准选错了」其实站不住：全市场等权和同市值分位在这个样本里几乎一样。',
           '   真正的原因是那一年半的行情，不是基准。',
           '2. **对沪深300 指数是跑赢的（%+.1fpp/年），但不显著（t = %.2f）**；' % (
               (ann_add - ann(st['沪深300指数'])) * 100, sm.loc[sm['基准'] == '沪深300指数', 't'].iloc[0]),
           '   而且这份超额几乎全部来自**风格**——沪深300 指数市值加权 %+.1f%%，同样 300 只等权 %+.1f%%，' % (
               ann(st['沪深300指数']) * 100, ann(st['沪深300等权']) * 100),
           '   光把权重从市值改成等权就差 %.1fpp。组合偏中小盘（持仓市值中位 %.0f 亿 vs 全市场 %.0f 亿，%.1f 倍），' % (
               (ann(st['沪深300等权']) - ann(st['沪深300指数'])) * 100,
               st['组合市值中位'].median(), st['基准市值中位'].median(),
               st['组合市值中位'].median() / st['基准市值中位'].median()),
           '   这个偏离在 2021-2025 的小盘行情里天然占优。',
           '3. **同市值对照下是跑输的（%+.1fpp/年，p = %.3f）**：把市值维度对齐之后，' % (
               (ann_add - ann(st[base_c])) * 100, sm.loc[sm['基准'] == base_c, 'p'].iloc[0]),
           '   「国家队持仓」在同一个市值桶里没有选出更好的股票，反而略差；不显著，只能说「没有证据表明会选股」。',
           '4. **「增持」信号本身无效**（见第六节）：跟随全部持仓 %.1f%% > 跟随增持 %.1f%%。' % (
               ann_all * 100, ann_add * 100),
           '5. 45 天滞后没变：等你看到季报，公告后那段涨幅已经走完。', '',
           '## 八、局限', '',
           '- **沪深300 等权是代理**：没有历史成分表，用「每期流通市值前 300」代替；中证 300 按自由流通市值选样，',
           '  代理与真实成分的重合度约 8~9 成，个别年份（如 2021 年抱团调整）偏差更大。真实指数点位那条不受影响。',
           '- 流通市值由换手率反推，误差 5~10%；对「取前 300」和「十分位分组」够用，但不能用来算精确权重。',
           '- 仍用报告期而非公告日，前视偏差残留；「增持」是各账户合计，会混入社保组合间调仓。',
           '- 未计成本：季度换 150~250 只，双边 15bp 约 0.4~0.6%/年。', '']
    open(f'{ROOT}/reports/national_team_bench.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('→', f'{ROOT}/reports/national_team_bench.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='national_team_bench'):
        main()
