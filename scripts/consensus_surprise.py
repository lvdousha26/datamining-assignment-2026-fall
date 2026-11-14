# -*- coding: utf-8 -*-
"""卖方一致预期偏离度：业绩预告 vs 分析师 FY2026 一致预期（点时可得）

背景
----
`ev_yjyg.md` 的结论是"业绩预告事件 5 日 CAR −0.58%、t −0.83，无超额"。一个自然的怀疑：
预告本身被提前消化，**超额应该藏在"公告 vs 预期"的偏离（surprise）里**，而不是预告方向里。

数据现实（先探过）
----
- iFinD 不可用；东财 `stock_research_report_em`（个股研报，按股票逐只拉）字段：
  日期 / 机构 / 东财评级 / 「2026/2027/2028-盈利预测-收益」（**EPS，元/股**）。
  年度列是**绝对年份**（2024 年的研报在 2026 列有值、2027 列全空），
  但接口只回传当前三个年份 → **只能构建 FY2026 一致预期**。
- 抽样 60 只预告股：公告日前 180 天内有 FY2026 预测的只有 23%——业绩预告股（多为业绩剧变的
  中小票）恰好是分析师覆盖最少的，**覆盖偏差必须写进结论**。
- 评估窗口落在 **2026 年（封存段）**：本脚本只做描述性证据，不作模型入库依据。

口径
----
- 一致预期 EPS26(t) = 公告日前 180 天内、日期 ≤ 公告日的研报 EPS 均值（≥2 家才取）。
- 股本 = FY2025 净利润 / FY2025 每股收益（yjbb_20251231）。
- 预告隐含全年净利润 = 预测数值 / (上年同期值 / FY2025 净利润)（用上年同期占比年化）。
- surprise = (隐含 − 一致预期净利润) / |一致预期净利润|，截尾 ±5。
- 事前收益：R 矩阵（daily_ret.npz，已清洗），t+1 起 5/20/60 日累计，超额 = 个股 − 全市场等权。

用法：python scripts/consensus_surprise.py [--fetch-only]
产出：reports/consensus_surprise.md + .csv；研报缓存 data/big/research/
"""
import glob
import io
import contextlib
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from figstyle import PALETTE, new_fig, save  # noqa: E402

ROOT = r'D:/quant'
RCACHE = f'{ROOT}/data/big/research'
os.makedirs(RCACHE, exist_ok=True)
LOOKBACK = 180
MIN_BROKERS = 2
HORIZONS = [5, 20, 60]


def fetch_all(codes):
    import akshare as ak
    t0 = time.time()
    n_ok = 0
    for k, code in enumerate(codes):
        f = f'{RCACHE}/{code}.parquet'
        if os.path.exists(f):
            n_ok += 1
            continue
        d = None
        with contextlib.redirect_stderr(io.StringIO()):   # 屏蔽 akshare 内部 tqdm，别把日志刷爆
            try:
                d = ak.stock_research_report_em(symbol=code)
            except KeyError:
                # akshare 对「无任何研报」的股票抛 KeyError 'infoCode'，属正常无覆盖，
                # 不要重试（原先 3 次重试 + 4.5s sleep 让每只白等 5 秒）
                d = pd.DataFrame()
            except Exception:
                time.sleep(0.4)
                try:
                    d = ak.stock_research_report_em(symbol=code)
                except Exception:
                    d = None
        if d is None or len(d) == 0:
            # 哨兵：无研报的股票也写空文件，断点续跑时不再重复请求
            pd.DataFrame({'date': [], 'broker': [], 'rating': [], 'eps26': []}).to_parquet(f)
            continue
        keep = d[['日期', '机构', '东财评级', '2026-盈利预测-收益']].copy()
        keep.columns = ['date', 'broker', 'rating', 'eps26']
        keep['date'] = pd.to_datetime(keep['date'])
        keep['eps26'] = pd.to_numeric(keep['eps26'], errors='coerce')
        keep.to_parquet(f)                              # 有覆盖/无覆盖都落盘
        if keep['eps26'].notna().any():
            n_ok += 1
        if (k + 1) % 200 == 0:
            print('  研报 %d/%d（累计 %d 只，%.0fs）' % (k + 1, len(codes), n_ok, time.time() - t0), flush=True)
    print('  研报缓存 %d 只 / %.0fs' % (n_ok, time.time() - t0), flush=True)
    return n_ok


def load_events():
    fs = [f for f in glob.glob(f'{ROOT}/data/big/yjyg/*.parquet') if '2026' in f]
    rows = []
    for f in fs:
        period = os.path.basename(f).replace('yjyg_', '').replace('.parquet', '')
        d = pd.read_parquet(f)
        d['period'] = period
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df['公告日期'] = pd.to_datetime(df['公告日期'])
    # 一个（股票, 报告期）只取最早一次预告（修正公告剔除，避免同事件重复计）
    df = df.sort_values('公告日期').groupby(['股票代码', 'period'], as_index=False).first()
    return df


def main():
    print('[1] 事件与研报', flush=True)
    ev = load_events()
    codes = sorted(ev['股票代码'].unique())
    print('  2026 预告事件 %d 个 / %d 只股票' % (len(ev), len(codes)), flush=True)
    cached = len([f for f in os.listdir(RCACHE) if f.endswith('.parquet')])
    if cached < len(codes) * 0.9:
        fetch_all(codes)

    print('[2] 基础财报（股本 / FY2025 净利润）', flush=True)
    yj = pd.read_parquet(f'{ROOT}/data/big/fin/yjbb_20251231.parquet')
    yj = yj[['股票代码', '每股收益', '净利润-净利润']].copy()
    yj.columns = ['code', 'eps25', 'np25']
    yj['shares'] = np.where(np.abs(yj['eps25']) > 1e-6, yj['np25'] / yj['eps25'], np.nan)
    base = dict(zip(yj['code'], zip(yj['np25'], yj['shares'])))

    print('[3] 一致预期与 surprise', flush=True)
    recs = []
    for _, e in ev.iterrows():
        code = e['股票代码']
        f = f'{RCACHE}/{code}.parquet'
        if not os.path.exists(f):
            continue
        r = pd.read_parquet(f)
        r = r[(r['date'] <= e['公告日期']) &
              (r['date'] >= e['公告日期'] - pd.Timedelta(days=LOOKBACK)) &
              r['eps26'].notna()]
        if len(r) < MIN_BROKERS:
            continue
        np25, shares = base.get(code, (np.nan, np.nan))
        if not (np.isfinite(np25) and np.isfinite(shares) and np25 > 0 and shares > 0):
            continue
        pred, prev = e['预测数值'], e['上年同期值']
        if not (np.isfinite(pred) and np.isfinite(prev)):
            continue
        share = prev / np25                       # 上年同期占全年比例
        if not (0.15 <= share <= 0.95):
            continue
        cons_np = r['eps26'].mean() * shares      # 一致预期 FY2026 净利润
        if not (np.isfinite(cons_np) and abs(cons_np) > 1e6):
            continue
        implied = pred / share                    # 预告隐含 FY2026 净利润
        surprise = np.clip((implied - cons_np) / abs(cons_np), -5, 5)
        recs.append(dict(code=code, period=e['period'], ann=e['公告日期'],
                         n_broker=len(r), cons_np=cons_np, implied=implied,
                         surprise=surprise, ytype=e['预告类型']))
    S = pd.DataFrame(recs)
    print('  匹配到一致预期的事件 %d 个（覆盖率 %.0f%%）' % (len(S), 100 * len(S) / len(ev)), flush=True)
    S.to_csv(f'{ROOT}/reports/consensus_surprise.csv', index=False)

    print('[4] 事前收益（R 矩阵）', flush=True)
    z = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    R, dates = z['R'], pd.to_datetime(pd.Series(z['dates']).astype(str))
    syms = [str(s).zfill(6) for s in z['symbols']]
    sidx = {s: i for i, s in enumerate(syms)}

    def fwd_ret(code, ann, h):
        i = sidx.get(code)
        t0 = int(dates.searchsorted(ann, side='left'))  # 公告日常非交易日 → 取之后首个交易日
        if i is None or t0 >= len(dates) - 2:
            return np.nan, np.nan
        sl = slice(t0 + 1, min(t0 + 1 + h, len(dates)))
        r = R[sl, i]
        if len(r) < 3 or np.isfinite(r).sum() < max(3, len(r) // 3):
            return np.nan, np.nan
        # 市场等权：当日全市场均值（只用同窗口）
        m = np.nanmean(R[sl], axis=1)
        cr = np.nancumprod(1 + np.nan_to_num(r))[-1] - 1
        cm = np.prod(1 + np.nan_to_num(m)) - 1
        return cr - cm, cr

    for h in HORIZONS:
        S[f'ex{h}'] = [fwd_ret(c, a, h)[0] for c, a in zip(S['code'], S['ann'])]

    print('[5] 分组', flush=True)
    S['q'] = pd.qcut(S['surprise'], 5, labels=False, duplicates='drop')
    rows = []
    for q, g in S.groupby('q'):
        row = dict(分位='Q%d' % (q + 1), n=len(g), surprise均值=g['surprise'].mean())
        for h in HORIZONS:
            row['ex%d' % h] = g[f'ex{h}'].mean()
        rows.append(row)
    T = pd.DataFrame(rows)
    print(T.round(4).to_string(index=False), flush=True)
    q1, q5 = S[S['q'] == 0], S[S['q'] == S['q'].max()]
    tests = {}
    for h in HORIZONS:
        a, b = q5[f'ex{h}'].dropna(), q1[f'ex{h}'].dropna()
        t = sps.ttest_ind(a, b, equal_var=False)
        tests[h] = (a.mean() - b.mean(), t.statistic, t.pvalue)
        print('  ex%d: Q5−Q1 = %+.2f%%（t=%.2f, p=%.3f）' % (h, 100 * tests[h][0], t.statistic, t.pvalue), flush=True)

    # 预告类型 × surprise 符号：方向被消化了吗
    S['surprise_sign'] = np.sign(S['surprise'])
    cross = S.groupby(['ytype', 'surprise_sign'])[f'ex{HORIZONS[1]}'].mean().unstack()
    print('\n预告类型 × surprise符号（ex%d 均值）：' % HORIZONS[1], flush=True)
    print((cross * 100).round(2).to_string(), flush=True)
    ctab = cross.mul(100).round(2)
    ctab.columns = ['低于预期' if c < 0 else ('高于预期' if c > 0 else '持平') for c in ctab.columns]
    ctab_lines = ['| 预告类型 | ' + ' | '.join(ctab.columns) + ' |',
                  '|---|' + '---|' * len(ctab.columns)]
    for idx, row in ctab.iterrows():
        ctab_lines.append('| %s | %s |' % (
            idx, ' | '.join('%+.2f' % v if np.isfinite(v) else '—' for v in row.values)))

    # ---- 图 ----
    fig, ax = new_fig(1, 1, w=8, h=4.2)
    xs = np.arange(len(T))
    wdt = 0.25
    for k, h in enumerate(HORIZONS):
        ax.bar(xs + (k - 1) * wdt, T['ex%d' % h] * 100, wdt, color=PALETTE[k], label='%d 日' % h)
    ax.set_xticks(xs)
    ax.set_xticklabels(T['分位'])
    ax.set_ylabel('公告后累计超额收益（%，全市场等权基准）')
    ax.set_xlabel('surprise 五分位（Q1 最低 = 公告远逊于一致预期）')
    ax.set_title('业绩预告 vs 卖方一致预期：偏离度分组的事后超额')
    ax.axhline(0, color='#888888', lw=0.8)
    ax.legend(frameon=False)
    save(fig, f'{ROOT}/reports/consensus_surprise.png',
         note='数据来源：东财个股研报（akshare stock_research_report_em，FY2026 EPS 预测，点时取公告日前 180 天）'
              '× 业绩预告 91,126 条中的 2026 年事件；评估窗口为 2026 年（封存段），仅描述性证据')

    # ---- 报告 ----
    md = ['# 卖方一致预期偏离度：业绩公告 vs 预期', '',
          '> %s｜承接 `ev_yjyg.md`「预告本身无超额」的结论，检验超额是否藏在 surprise 里' %
          pd.Timestamp.today().strftime('%Y-%m-%d'), '',
          '## 口径', '',
          '| 项 | 设定 |', '|---|---|',
          '| 一致预期 | 东财个股研报的 FY2026 EPS 预测，公告日前 180 天内、≥2 家机构，点时可得（无前视） |',
          '| 预期净利润 | 一致 EPS × 股本（股本 = FY2025 净利润 / FY2025 EPS） |',
          '| 预告隐含全年 | 预测数值 / (上年同期值 / FY2025 净利润)——用上年同期占比年化 |',
          '| surprise | (隐含 − 一致预期) / |一致预期|，截尾 ±5 |',
          '| 事后收益 | `daily_ret.npz` 清洗后 R 矩阵，公告次日起重仓持有 5/20/60 日，减全市场等权 |', '',
          '## 覆盖率（关键局限）', '',
          '2026 年预告事件 %d 个（%d 只股票），匹配到一致预期的只有 **%d 个（%.0f%%）**。'
          '抽样探得的整体覆盖率约 23%%：**业绩预告股（业绩剧变的中小票）恰是分析师覆盖最少的**——'
          '本研究的样本天然偏向机构重仓的大票，结论不能外推到全市场。' % (len(ev), ev['股票代码'].nunique(), len(S), 100 * len(S) / len(ev)), '',
          '## 分组结果', '',
          '| 分位 | n | surprise 均值 | 5 日超额 | 20 日超额 | 60 日超额 |', '|---|---|---|---|---|---|']
    for _, r in T.iterrows():
        md.append('| %s | %d | %+.2f | %+.2f%% | %+.2f%% | %+.2f%% |' % (
            r['分位'], r['n'], r['surprise均值'],
            100 * r['ex5'], 100 * r['ex20'], 100 * r['ex60']))
    for h, (diff, tstat, pv) in tests.items():
        md.append('')
        md.append('**%d 日超额 Q5−Q1 = %+.2f%%（t=%.2f，p=%.3f）**' % (h, 100 * diff, tstat, pv))
    md += ['', '## 预告类型 × surprise 符号（20 日超额，%）', ''] + ctab_lines + ['',
           '## 必须写的三条警示', '',
           '1. **评估窗口是 2026 年（封存段）**：本项目 lockbox 已一次性消耗，本表任何数字都只能作为'
           '**描述性证据**，不能用于模型选股/入库决策；',
           '2. **覆盖偏差 23%**：有预期覆盖的事件集中在机构关注的标的，surprise 若有效也可能是'
           '"大票对预期更敏感"的另一种写法；',
           '3. **年化假设**：预告是期内累计值（Q1/H1/Q3），用上年同期占比折全年是粗近似——'
           '全年业绩节奏变化大的公司（如四季度集中确认）会失真。', '']
    open(f'{ROOT}/reports/consensus_surprise.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n→', f'{ROOT}/reports/consensus_surprise.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='consensus'):
        if '--fetch-only' in sys.argv:
            ev = load_events()
            fetch_all(sorted(ev['股票代码'].unique()))
        else:
            main()
