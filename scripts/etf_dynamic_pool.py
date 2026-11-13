# -*- coding: utf-8 -*-
"""ETF 三因子动量轮动 · 动态池版

在 etf_momentum3.py（固定 4 只池）基础上，把股票池换成「当期成立满 2 年、规模前 20」的动态池。

设计
----
- 名单：同花顺 ETF 全表 `fund_etf_category_ths`（当前 1,783 只，**剔除货币型 25 只**）。
  注意这是**当前名单**，已清盘 ETF 不在其中 → 仍存在幸存者偏差（2017-2018 头部的分级基金已被
  监管清算、不在表里），这是数据源限制，报告里如实说明。
- 行情：新浪 `fund_etf_hist_sina`（东财被限流），含 amount 列，缓存到 `data/big/etf_all/`。
- 「规模」的口径：**历史规模拿不到**（上交所只有当月份额快照、深交所接口损坏），用
  **60 日日均成交额**做规模代理——点时可得、无前视；大基金成交额基本也大，但两者不恒等。
- 「成立满 2 年」：用行情首日近似上市日（误差 < 1 周）。
- 调仓：每 5 个交易日（与原文版一致），每期重算池 + 三因子打分（0.3/0.3/0.4），
  阈值 1.5（挑战者 > 持仓者 1.5 倍才换），单边成本 5bp。
- 对照：①动态池三因子选 1；②动态池 top20 等权（同频率再平衡）；③原固定 4 池三因子
  （2020-07 起，515100 上市）。

用法：python scripts/etf_dynamic_pool.py   [--dl-only]
产出：reports/etf_dynamic_pool.md / .png / etf_pool_hist.csv
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
CACHE = f'{ROOT}/data/big/etf_all'
os.makedirs(CACHE, exist_ok=True)
BIAS_N, MOM_DAY, SLOPE_N = 25, 25, 25
W = (0.3, 0.3, 0.4)
THR = 1.5
COST = 0.0005
STEP = 5
POOL_N = 20
MIN_AGE = 365 * 2              # 成立满 2 年（自然日）
FIXED_POOL = {'515100': '红利低波', '159949': '创业板50', '513100': '纳指', '518880': '黄金'}


def exchange(code):
    return 'sh' if code[:2] in ('51', '52', '53', '55', '56', '58') else 'sz'


def universe():
    """同花顺 ETF 全表，剔除货币型，返回 [(code, name)]"""
    import akshare as ak
    d = ak.fund_etf_category_ths()
    d = d[d['基金类型'] != '货币型']
    codes = d['基金代码'].astype(str).str.zfill(6)
    return list(zip(codes, d['基金名称']))


def download(codes):
    n_ok = 0
    t0 = time.time()
    for k, (code, name) in enumerate(codes):
        f = f'{CACHE}/{code}.parquet'
        if os.path.exists(f):
            n_ok += 1
            continue
        import akshare as ak
        d = None
        for r in range(3):
            try:
                d = ak.fund_etf_hist_sina(symbol=f'{exchange(code)}{code}')
                break
            except Exception:
                time.sleep(1 + r)
        if d is None or len(d) == 0:
            continue
        d['date'] = pd.to_datetime(d['date'])
        d = d.set_index('date')[['open', 'high', 'low', 'close', 'volume', 'amount']].astype(float)
        if len(d) >= 60:
            d.to_parquet(f)
            n_ok += 1
        if (k + 1) % 100 == 0:
            print('  下载 %d/%d（累计 %d 只，%.0fs）' % (k + 1, len(codes), n_ok, time.time() - t0), flush=True)
    print('  行情缓存 %d 只 / %.0fs' % (n_ok, time.time() - t0), flush=True)
    return n_ok


def load_all():
    out = {}
    for f in os.listdir(CACHE):
        if f.endswith('.parquet'):
            out[f[:6]] = pd.read_parquet(f'{CACHE}/{f}')
    return out


def three_factors(sub):
    """一段行情（≤61 行）算三因子，返回 (f1, f2, f3) 或 None"""
    c = sub['close']
    if len(c) < MOM_DAY:
        return None
    bias = c / c.rolling(BIAS_N, min_periods=1).mean()
    br = bias.iloc[-MOM_DAY:]
    if len(br) < MOM_DAY or br.iloc[0] == 0:
        return None
    y = (br / br.iloc[0]).values
    f1 = np.polyfit(np.arange(len(y)), y, 1)[0] * 10000
    w = c.iloc[-SLOPE_N:]
    yy = (w / w.iloc[0]).values
    xx = np.arange(1, len(yy) + 1)
    k, b = np.polyfit(xx, yy, 1)
    pred = k * xx + b
    r2 = 1 - ((yy - pred) ** 2).sum() / max(((yy - yy.mean()) ** 2).sum(), 1e-12)
    f2 = 10000 * k * max(r2, 0)
    pivot = (sub['open'] + sub['high'] + sub['low'] + sub['close']) / 4
    p0, p1 = pivot.iloc[0], pivot.iloc[-1]
    if p0 <= 0 or p1 <= 0:
        return None
    mom = 100 * np.log(p1 / p0)
    lp = np.log(pivot)
    vol = lp.diff().abs().sum()
    er = abs(lp.iloc[-1] - lp.iloc[0]) / vol if vol > 0 else 0
    return (f1, f2, mom * er)


def zscore(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def stat(nav):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    return dict(年化=ann, 波动=vol, 夏普=(ann - 0.02) / vol if vol > 0 else np.nan,
                最大回撤=(nav / nav.cummax() - 1).min(), 总收益=nav.iloc[-1] - 1, 年数=yrs)


def run_dynamic(px, liq60, first_day, cal, start):
    """动态池三因子选 1；返回 (nav, 池历史 DataFrame)"""
    idx = [i for i, t in enumerate(cal) if t >= start]
    hold, nav, navs, hist = None, 1.0, [], []
    prev_c = {}
    for k, i in enumerate(idx):
        t = cal[i]
        if hold is not None:
            d = px[hold]
            if t in d.index and hold in prev_c and prev_c[hold] > 0:
                nav *= d.loc[t, 'close'] / prev_c[hold]
        if k % STEP == 0:
            elig = []
            for s in px:
                if first_day.get(s, pd.Timestamp.max) > t - pd.Timedelta(days=MIN_AGE):
                    continue
                l = liq60[s].get(t)
                if l is not None and np.isfinite(l) and l > 0:
                    elig.append((l, s))
            pool = [s for _, s in sorted(elig, reverse=True)[:POOL_N]]
            hist.append(dict(date=t, pool=';'.join(pool), hold=hold))
            fs = {}
            for s in pool:
                f = three_factors(px[s].loc[:t].tail(61))
                if f is not None:
                    fs[s] = f
            if len(fs) >= 2:
                m = np.array(list(fs.values()), float)
                tot = W[0] * zscore(m[:, 0]) + W[1] * zscore(m[:, 1]) + W[2] * zscore(m[:, 2])
                sc = dict(zip(fs.keys(), tot))
                best = max(sc, key=sc.get)
                if hold is None or sc[best] > THR * sc.get(hold, -1e9):
                    if best != hold and hold is not None:
                        nav *= (1 - COST)
                    hold = best
        prev_c = {s: px[s].loc[t, 'close'] for s in px if t in px[s].index}
        navs.append(nav)
    return pd.Series(navs, index=cal[idx]), pd.DataFrame(hist)


def run_dynamic_ew(px, liq60, first_day, cal, start):
    """动态池 top20 等权，同频率再平衡，成本按单边换手 × 2 × 5bp"""
    idx = [i for i, t in enumerate(cal) if t >= start]
    nav, navs, weights = 1.0, [], None
    prev_px = {}
    for k, i in enumerate(idx):
        t = cal[i]
        if weights:
            r = 0.0
            for s, w in weights.items():
                d = px[s]
                if t in d.index and s in prev_px and prev_px[s] > 0:
                    r += w * (d.loc[t, 'close'] / prev_px[s] - 1)
            nav *= (1 + r)
        if k % STEP == 0:
            elig = []
            for s in px:
                if first_day.get(s, pd.Timestamp.max) > t - pd.Timedelta(days=MIN_AGE):
                    continue
                l = liq60[s].get(t)
                if l is not None and np.isfinite(l) and l > 0:
                    elig.append((l, s))
            pool = [s for _, s in sorted(elig, reverse=True)[:POOL_N]]
            new_w = {s: 1 / len(pool) for s in pool} if pool else {}
            if weights:
                keys = set(weights) | set(new_w)
                turnover = 0.5 * sum(abs(new_w.get(s, 0) - weights.get(s, 0)) for s in keys)
                nav *= (1 - turnover * 2 * COST)
            weights = new_w
        prev_px = {s: px[s].loc[t, 'close'] for s in px if t in px[s].index}
        navs.append(nav)
    return pd.Series(navs, index=cal[idx])


def main():
    print('[1] 名单与行情', flush=True)
    cache_n = len([f for f in os.listdir(CACHE) if f.endswith('.parquet')])
    if cache_n < 1000:
        codes = universe()
        print('  THS 全表（剔货币）%d 只' % len(codes), flush=True)
        download(codes)
    px = load_all()
    print('  载入 %d 只行情' % len(px), flush=True)

    # 主日历：取 510300（沪深300ETF，最老最活跃）的日期
    cal = px['510300'].index
    # 60 日均成交额 + 首日
    liq60 = {s: d['amount'].rolling(60, min_periods=40).mean() for s, d in px.items()}
    first_day = {s: d.index[0] for s, d in px.items()}

    start = pd.Timestamp('2017-01-01')
    print('[2] 动态池回测（%s 起，每 %d 日评估）' % (start.date(), STEP), flush=True)
    nav_dyn, hist = run_dynamic(px, liq60, first_day, cal, start)
    nav_ew = run_dynamic_ew(px, liq60, first_day, cal, start)

    # 池内成分统计
    ps = hist['pool'].str.split(';')
    n_eval = len(ps)
    churn = [(len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)) for a, b in zip(ps[:-1], ps[1:])]
    ages = []
    for i in range(n_eval):
        a = [(hist['date'][i] - first_day.get(s, pd.Timestamp.now())).days / 365.25 for s in ps[i]]
        ages.append((min(a), np.mean(a), max(a)))
    age_min = np.mean([x[0] for x in ages]); age_mean = np.mean([x[1] for x in ages]); age_max = np.mean([x[2] for x in ages])

    print('[3] 固定 4 池对照', flush=True)
    fx = {c: px[c] for c in FIXED_POOL if c in px}
    fcal = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in fx.values()])))
    f_start = max(d.index[0] for d in fx.values())
    hold, nav, navs = None, 1.0, []
    prev_c = {}
    fidx = [i for i, t in enumerate(fcal) if t >= f_start]
    for k, i in enumerate(fidx):
        t = fcal[i]
        if hold is not None and t in fx[hold].index and prev_c.get(hold, 0) > 0:
            nav *= fx[hold].loc[t, 'close'] / prev_c[hold]
        if k % STEP == 0:
            fs = {s: three_factors(fx[s].loc[:t].tail(61)) for s in fx}
            fs = {s: f for s, f in fs.items() if f is not None}
            if len(fs) >= 2:
                m = np.array(list(fs.values()), float)
                tot = W[0] * zscore(m[:, 0]) + W[1] * zscore(m[:, 1]) + W[2] * zscore(m[:, 2])
                sc = dict(zip(fs.keys(), tot))
                best = max(sc, key=sc.get)
                if hold is None or sc[best] > THR * sc.get(hold, -1e9):
                    if best != hold and hold is not None:
                        nav *= (1 - COST)
                    hold = best
        prev_c = {s: fx[s].loc[t, 'close'] for s in fx if t in fx[s].index}
        navs.append(nav)
    nav_fix = pd.Series(navs, index=fcal[fidx])

    rows = []
    for name, nav in [('动态池 top20 · 三因子选1', nav_dyn),
                      ('动态池 top20 · 等权', nav_ew),
                      ('固定 4 池 · 三因子（原文）', nav_fix)]:
        st = stat(nav)
        st['策略'] = name
        rows.append(st)
    df = pd.DataFrame(rows)[['策略', '年化', '波动', '夏普', '最大回撤', '总收益', '年数']]
    print('\n' + df.round(4).to_string(index=False), flush=True)
    hist.to_csv(f'{ROOT}/reports/etf_pool_hist.csv', index=False)

    # ---- 图 ----
    fig, ax = new_fig(1, 1, w=8, h=4.2)
    for k, (nm, nav) in enumerate([('动态池 top20 · 三因子选1', nav_dyn),
                                   ('动态池 top20 · 等权', nav_ew),
                                   ('固定 4 池 · 三因子（原文）', nav_fix)]):
        ax.plot(nav.index, nav.values, color=PALETTE[k], lw=1.8, label=nm)
    ax.set_yscale('log')
    ax.set_ylabel('累计净值（倍，对数坐标）'); ax.set_xlabel('日期（年）')
    ax.set_title('ETF 三因子动量轮动：动态池 vs 固定池（动态池 2017 起 / 固定池 2020-07 起）')
    ax.legend(frameon=False)
    save(fig, f'{ROOT}/reports/etf_dynamic_pool.png',
         note='数据来源：新浪财经 ETF 日线（akshare fund_etf_hist_sina，前复权）+ 同花顺 ETF 名单；'
              '池=成立满 2 年且 60 日均成交额前 20（剔货币型）；每 5 日评估，单边 5bp')

    # ---- 报告 ----
    md = ['# ETF 三因子动量轮动 · 动态池版', '',
          '> %s｜承接 `etf_momentum3.md`：把固定 4 只池换成「当期成立满 2 年、规模前 20」动态池' %
          pd.Timestamp.today().strftime('%Y-%m-%d'), '',
          '## 口径', '',
          '| 项 | 设定 |', '|---|---|',
          '| 名单 | 同花顺 ETF 全表 1,783 只，**剔除货币型 25 只** |',
          '| 成立满 2 年 | 行情首日近似上市日（自然日 730 天） |',
          '| 规模前 20 | **60 日日均成交额**做规模代理（历史规模拿不到：上交所只有当月份额快照、深交所接口损坏；成交额点时可得、无前视） |',
          '| 因子 / 权重 / 阈值 / 成本 | 与原文复现完全一致：三因子 0.3/0.3/0.4，挑战者 > 1.5 倍才换，单边 5bp，每 5 个交易日评估 |',
          '| 对照 | ①动态池三因子选 1；②动态池 top20 等权；③固定 4 池（原文，2020-07 起） |', '',
          '## 结果', '',
          '| 策略 | 年化 | 波动 | 夏普 | 最大回撤 | 总收益 | 区间 |', '|---|---|---|---|---|---|---|']
    for _, r in df.iterrows():
        md.append('| %s | **%.1f%%** | %.1f%% | %.2f | %.1f%% | %.1f 倍 | %.1f 年 |' % (
            r['策略'], r['年化'] * 100, r['波动'] * 100, r['夏普'], r['最大回撤'] * 100,
            1 + r['总收益'], r['年数']))
    md += ['', '## 动态池本身的画像（%d 次评估）' % n_eval, '',
           '- 池成分相邻两期 Jaccard 重合度均值 **%.2f**（=1 表示从不换血，越低换手越高）；' % np.mean(churn),
           '- 池内成员平均年龄 **%.1f 年**（最年轻均值 %.1f / 最老均值 %.1f，"满 2 年"规则生效中）；' % (
               age_mean, age_min, age_max),
           '- 池历史全表见 `etf_pool_hist.csv`（每次评估的 20 只 + 当时持仓）。', '',
           '## 幸存者偏差：比原文小得多，但不是零', '',
           '1. 名单是**当前**的 THS 全表——已清盘 ETF 不在其中。2017-2018 年成交额头部其实是**分级基金**'
           '（150 开头，2018-2020 被监管清算），它们被系统性排除在动态池之外。所以早期动态池更接近'
           '"现存宽基/行业 ETF 池"，不是当时真实的全 ETF 池；',
           '2. 但与原文「事后手挑 4 只」相比，动态池的选择规则是**事前可执行**的（成立满 2 年 + 成交额前 20），'
           '不含"我知道纳指会涨所以放进去"的信息——这是本次对照里最重要的改进；',
           '3. 规模代理用成交额而非真实规模：大赎回期间份额骤降但成交额可能反而放大（恐慌交易），'
           '两者在极端期会短暂背离。', '',
           '## 结论', '',
           '（见上表：动态池三因子相对固定池和等权的差值——数字说话，不预设方向。）', '']
    open(f'{ROOT}/reports/etf_dynamic_pool.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(md))
    print('\n→', f'{ROOT}/reports/etf_dynamic_pool.md')


if __name__ == '__main__':
    with RunLogger(__file__, tag='etf_dyn'):
        if '--dl-only' in sys.argv:
            codes = universe()
            print('THS 全表（剔货币）%d 只' % len(codes), flush=True)
            download(codes)
        else:
            main()
