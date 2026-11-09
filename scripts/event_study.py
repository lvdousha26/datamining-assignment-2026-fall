# -*- coding: utf-8 -*-
"""通用事件研究框架（Event Study）

给它一个 (股票, 日期, 事件描述) 列表，自动输出：
    ① 逐事件 CAR 表（多窗口，超额 = 个股 vs 基准的几何差）
    ② 安慰剂分位（该股全部历史同长度窗口里，事件 CAR 排第几百分位）
    ③ 同侪排序（同行业 / 手工 peer 组里，事件股的超额排第几）
    ④ 汇总统计（CAAR、跨事件 t 值、胜率、安慰剂中位分位，按事件类型分组）

用法
    python scripts/event_study.py --events events/demo_cn.csv --market cn
    python scripts/event_study.py --events events/demo_us.csv --market us
    python scripts/event_study.py --events my.csv --market cn --windows 1,5,20,60 --out reports/ev_x

事件表 CSV（必需 symbol、date；可选 label、note）
    symbol,date,label,note
    601155,2019-07-03,实控人丑闻,王振华被拘
    000001,2020-02-03,财报超预期,净利润同比+30%

输出（--out 决定前缀，默认 reports/event_study）
    <out>.csv            逐事件 × 窗口的明细
    <out>_summary.csv    按 label 汇总
    <out>_peers.csv      同侪排序
    <out>.md             文字报告
    <out>.png            CAR 曲线 + 安慰剂分布

口径说明
    CAR = (1+r_stock) / (1+r_bench) - 1      几何超额，不是简单相减
    安慰剂 = 同一只股票历史上所有同长度窗口，不抽样、用全样本
    t 值   = 跨事件（事件间独立假设），不是跨时间
    事件日 = 若落在非交易日，取 **下一个** 交易日（--align prev 可改为前一个）
"""
import argparse
import json
import os
import sys

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
CACHE_US = f'{ROOT}/data/us'
DEFAULT_WINDOWS = [1, 5, 20, 60, 250]


# ---------------------------------------------------------------- 市场抽象
class Market:
    """统一的价格 / 基准 / 同侪接口。价格一律用净值（cumprod(1+r)），
    这样 A 股可以直接吃已清洗过的日收益矩阵，不必碰复权价。"""

    name = 'base'

    def net(self, symbol):
        """单只股票的净值序列（index=日期）"""
        raise NotImplementedError

    def bench(self):
        """基准净值序列"""
        raise NotImplementedError

    def peers(self, symbol, date):
        """与 symbol 在 date 同组的其他股票代码"""
        raise NotImplementedError

    def resolve_date(self, date, align='next'):
        """把事件日映射到交易日"""
        idx = self.index()
        i = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(date))))
        if align == 'prev':
            i = max(i - 1, 0)
        elif i >= len(idx):
            i = len(idx) - 1
        return idx[i], i

    def index(self):
        raise NotImplementedError


class CNMarket(Market):
    """A 股：直接用 data/big/daily_ret.npz 的日收益矩阵
    （market_verify.py 产出，已剔除停牌、复权断层、|r|>50% 的脏值）"""

    name = 'cn'

    def __init__(self, bench='ew'):
        d = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
        self.R = d['R']
        self._dates = pd.to_datetime(pd.Series(d['dates']).astype(str))
        self.syms = [str(s).zfill(6) for s in d['symbols']] if 'symbols' in d.files else None
        if self.syms is None:
            z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
            self.syms = [str(s).zfill(6) for s in z['symbols']]
        self.col = {s: j for j, s in enumerate(self.syms)}
        self._net = {}
        self.bench_mode = bench
        # 行业映射（申万一级，用于同侪）
        l1 = pd.read_parquet(f'{ROOT}/data/big/sw_l1.parquet')
        l1['symbol'] = l1['symbol'].astype(str).str.zfill(6)
        self.ind = dict(zip(l1['symbol'], l1['l1']))
        self._by_ind = {}
        for s, g in l1.groupby('l1'):
            self._by_ind[g['l1'].iloc[0]] = [x for x in g['symbol'] if x in self.col]

    def index(self):
        return self._dates

    def net(self, symbol):
        s = str(symbol).zfill(6)
        if s in self._net:
            return self._net[s]
        if s not in self.col:
            raise KeyError(f'{s} 不在日收益矩阵里')
        r = self.R[:, self.col[s]].astype(np.float64)
        r = np.nan_to_num(r, nan=0.0)
        v = pd.Series(np.cumprod(1 + r), index=self._dates)
        self._net[s] = v
        return v

    def bench(self):
        if self.bench_mode == 'csi':
            idx = pd.read_pickle(f'{ROOT}/data/big/idx_daily.pkl')['中证全指']
            idx.index = pd.to_datetime(idx.index)
            return idx.reindex(self._dates).ffill()
        # 全市场等权（与 Y2 定义一致）
        v = pd.Series(np.cumprod(1 + np.nan_to_num(np.nanmean(self.R, 1), nan=0.0)), index=self._dates)
        return v

    def peers(self, symbol, date):
        s = str(symbol).zfill(6)
        ind = self.ind.get(s)
        if ind is None:
            return []
        return [x for x in self._by_ind.get(ind, []) if x != s]


class USMarket(Market):
    """美股：akshare 后复权（hfq）。qfq 在东财美股接口上会产生负价格，不能用。"""

    name = 'us'
    BENCH = 'IXIC'

    def __init__(self, peer_file=None):
        os.makedirs(CACHE_US, exist_ok=True)
        self._px = {}
        self._mapf = f'{CACHE_US}/_prefix.json'
        self.pmap = json.load(open(self._mapf)) if os.path.exists(self._mapf) else {}
        self.peer_file = peer_file
        self._custom_peers = None
        if peer_file and os.path.exists(peer_file):
            self._custom_peers = [x.strip() for x in open(peer_file) if x.strip()]

    def index(self):
        return self._load(self.BENCH).index

    def _resolve(self, tk):
        if tk in self.pmap:
            return self.pmap[tk]
        import akshare as ak
        for p in ['105', '106', '107']:
            try:
                d = ak.stock_us_hist(symbol=f'{p}.{tk}', period='daily',
                                     start_date='20260101', end_date='20261231', adjust='hfq')
                if len(d):
                    self.pmap[tk] = f'{p}.{tk}'
                    json.dump(self.pmap, open(self._mapf, 'w'))
                    return self.pmap[tk]
            except Exception:
                pass
        raise RuntimeError('无法解析美股代码 ' + tk)

    def _load(self, tk):
        if tk in self._px:
            return self._px[tk]
        f = f'{CACHE_US}/{tk}_hfq.parquet'
        if os.path.exists(f):
            s = pd.read_parquet(f)['close']
        else:
            import akshare as ak
            if tk == self.BENCH:
                d = ak.index_us_stock_sina(symbol='.IXIC')
                d['date'] = pd.to_datetime(d['date'])
                s = d.set_index('date')['close'].sort_index().dropna()
            else:
                d = ak.stock_us_hist(symbol=self._resolve(tk), period='daily',
                                     start_date='20040101', end_date='20261231', adjust='hfq')
                d['date'] = pd.to_datetime(d['日期'])
                s = d.set_index('date')['收盘'].astype(float).sort_index().dropna()
            assert s.min() > 0, f'{tk} 出现非正价格，复权数据异常'
            s.to_frame('close').to_parquet(f)
        self._px[tk] = s
        return s

    def net(self, symbol):
        return self._load(symbol)

    def bench(self):
        return self._load(self.BENCH)

    def peers(self, symbol, date):
        if self._custom_peers:
            return [x for x in self._custom_peers if x != symbol]
        return []


# ---------------------------------------------------------------- 核心计算
def car_at(net_s, net_b, i0, w):
    """事件日后第 w 个交易日的几何超额 CAR；越界返回 nan"""
    if i0 + w >= len(net_s) or i0 + w < 0:
        return np.nan
    r = net_s.iloc[i0 + w] / net_s.iloc[i0] - 1
    rb = net_b.iloc[i0 + w] / net_b.iloc[i0] - 1
    return (1 + r) / (1 + rb) - 1


def placebo_pct(net_s, net_b, w, actual, exclude=None, exclude_win=10):
    """安慰剂：该股历史上所有长度为 w 的窗口 CAR，看实际值排第几百分位。
    exclude=(i0) 时剔除事件日附近 ±exclude_win 的窗口，避免自己和自己比。"""
    s = net_s.values
    b = net_b.reindex(net_s.index).values
    if len(s) <= w + 10:
        return np.nan, np.nan, 0
    rs = s[w:] / s[:-w] - 1
    rb = b[w:] / b[:-w] - 1
    cars = (1 + rs) / (1 + rb) - 1
    ok = np.isfinite(cars)
    if exclude is not None:
        pos = np.arange(len(cars))
        ok &= (np.abs(pos - exclude) > exclude_win)
    cars = cars[ok]
    if len(cars) < 30:
        return np.nan, np.nan, len(cars)
    pct = (cars < actual).mean() * 100
    return pct, float(np.median(cars)), len(cars)


def run_one(mkt, symbol, date, windows, align='next', lag=0, peer_win=None):
    """单个事件的完整结果"""
    try:
        net_s = mkt.net(symbol)
    except Exception as e:
        return dict(symbol=symbol, error=str(e)[:60])
    net_b = mkt.bench().reindex(net_s.index).ffill()
    d0, _ = mkt.resolve_date(date, align)
    i0 = int(np.searchsorted(net_s.index.values, np.datetime64(d0))) + lag
    if i0 >= len(net_s):
        return dict(symbol=symbol, error='事件日超出数据范围')

    row = dict(symbol=str(symbol).zfill(6) if mkt.name == 'cn' else symbol,
               date=str(d0.date()), i0=i0)
    for w in windows:
        c = car_at(net_s, net_b, i0, w)
        pct, med, npl = placebo_pct(net_s, net_b, w, c, exclude=i0)
        row[f'car{w}'] = c
        row[f'pct{w}'] = pct
        row[f'pl_med{w}'] = med
    row['pl_n'] = npl

    # 同侪排序（用最长窗口）
    pw = peer_win or max(windows)
    peers = mkt.peers(symbol, d0) if mkt.name == 'cn' else mkt.peers(symbol, d0)
    if peers:
        vals = []
        for p in peers:
            try:
                ns = mkt.net(p)
            except Exception:
                continue
            nb = net_b.reindex(ns.index).ffill()
            i = int(np.searchsorted(ns.index.values, np.datetime64(d0)))
            if i + pw >= len(ns):
                continue
            v = car_at(ns, nb, i, pw)
            if np.isfinite(v):
                vals.append(v)
        if len(vals) >= 5:          # 美股手工 peer 组通常只有 10 个
            vals = np.array(vals)
            row[f'peer_n{pw}'] = len(vals)
            row[f'peer_med{pw}'] = float(np.median(vals))
            row[f'peer_rank{pw}'] = float((vals > row.get(f'car{pw}', np.nan)).mean() * 100)
    return row


def summarize(detail, windows):
    """按 label 汇总：CAAR、跨事件 t、胜率、安慰剂中位分位"""
    rows = []
    groups = [('ALL', detail)] if 'label' not in detail.columns else \
             [('ALL', detail)] + list(detail.groupby('label'))
    for name, g in groups:
        if len(g) == 0:
            continue
        row = dict(label=name, n=len(g))
        for w in windows:
            c = g[f'car{w}'].dropna()
            if len(c) == 0:
                continue
            row[f'car{w}_mean'] = c.mean()
            row[f'car{w}_med'] = c.median()
            row[f'car{w}_win'] = (c > 0).mean()
            row[f'car{w}_n'] = len(c)
            row[f'car{w}_t'] = c.mean() / (c.std(ddof=1) / np.sqrt(len(c))) if len(c) > 2 and c.std() > 0 else np.nan
            p = g[f'pct{w}'].dropna()
            row[f'pct{w}_med'] = p.median() if len(p) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _md_table(df, floatfmt='.3f'):
    """不依赖 tabulate 的 markdown 表格生成"""
    if df is None or len(df) == 0:
        return '_（无数据）_'
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |',
             '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append('—' if pd.isna(v) else format(v, floatfmt))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append('' if v is None else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def report(detail, summ, peers_df, mkt_name, windows, out):
    L = []
    L.append('# 事件研究结果\n')
    L.append(f'> 市场：{mkt_name}｜事件数：{len(detail)}｜窗口：{windows} 日｜'
             f'CAR = (1+个股)/(1+基准)−1（几何超额）\n')
    L.append('## 一、汇总（CAAR / t / 胜率 / 安慰剂中位分位）\n')
    hdr = '| 事件类型 | N |'
    sep = '|---|---|'
    for w in windows:
        hdr += f' CAR{w} 均值 | t | 胜率 | 安慰剂分位 |'
        sep += '---|---|---|---|'
    L.append(hdr)
    L.append(sep)
    for _, r in summ.iterrows():
        line = f"| {r['label']} | {int(r['n'])} |"
        for w in windows:
            cm = r.get(f'car{w}_mean', np.nan)
            tv = r.get(f'car{w}_t', np.nan)
            wn = r.get(f'car{w}_win', np.nan)
            pm = r.get(f'pct{w}_med', np.nan)
            f1 = lambda x: '—' if pd.isna(x) else f'{x * 100:+.1f}%'
            f2 = lambda x: '—' if pd.isna(x) else f'{x:.2f}'
            f3 = lambda x: '—' if pd.isna(x) else f'{x * 100:.0f}%'
            line += f' {f1(cm)} | {f2(tv)} | {f3(wn)} | {f3(pm / 100 if pd.notna(pm) else np.nan)} |'
        L.append(line)
    L.append('\n> 安慰剂分位中位数≈50% 说明「这个事件毫无特殊之处」；'
             '<10% 或 >90% 才算有统计意义上的异常。')

    # 各窗口有效样本数（长窗口会因数据末端不足而减少）
    L.append('\n**各窗口有效事件数**（低于 30 时统计功效不足，t 值只能当参考）\n')
    L.append('| 窗口 | ' + ' | '.join(f'{w} 日' for w in windows) + ' |')
    L.append('|---|' + '|'.join(['---'] * len(windows)) + '|')
    allrow = summ[summ['label'] == 'ALL']
    if len(allrow):
        r0 = allrow.iloc[0]
        L.append('| 有效 N | ' + ' | '.join(
            str(int(r0.get(f'car{w}_n', 0))) for w in windows) + ' |')
    L.append('')

    L.append('## 二、逐事件明细\n')
    cols = ['symbol', 'date', 'label'] + [f'car{w}' for w in windows if f'car{w}' in detail.columns]
    cols = [c for c in cols if c in detail.columns]
    L.append(_md_table(detail[cols]))
    L.append('')

    if peers_df is not None and len(peers_df):
        L.append('## 三、同侪排序（最长窗口）\n')
        d = peers_df.copy()
        for c in d.columns:
            if c.startswith('peer_rank'):
                d[c] = d[c].map(lambda x: f'{x:.1f}%' if pd.notna(x) else '—')
            elif c.startswith('peer_med'):
                d[c] = d[c].map(lambda x: f'{x * 100:+.1f}%' if pd.notna(x) else '—')
        L.append(_md_table(d))
        L.append('\n> peer_rank = 有多少比例的同行跑赢了事件股（越高=事件股越差）。'
                 'peer_n 很小时（美股手写 peer 组约 10 个）精度只有 10%，只能做定性参考。\n')

    L.append('## 四、怎么读这张表\n')
    L.append('1. **先看安慰剂分位**：单事件 250 日 CAR +30% 听起来很好，'
             '但若该股历史上有一半的 250 日窗口都能涨 30%，那就毫无意义。')
    L.append('2. **再看跨事件 t**：N 个事件里 CAR 均值 / 标准误。'
             'N<10 时基本没有统计功效，只能当案例。')
    L.append('3. **最后看同侪**：同行业同期也在涨，说明是行业 beta 而非事件效应。')
    L.append('')
    open(out, 'w', encoding='utf-8').write('\n'.join(L))


def plot_car(detail, summ, windows, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for fp in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']:
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
                break
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    w = [x for x in windows if f'car{x}' in detail.columns]
    mu = [detail[f'car{x}'].mean() * 100 for x in w]
    med = [detail[f'car{x}'].median() * 100 for x in w]
    ax[0].plot(w, mu, 'o-', color='#c0392b', label='均值 CAAR')
    ax[0].plot(w, med, 's--', color='#2980b9', label='中位数')
    ax[0].axhline(0, color='gray', lw=0.8)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('窗口（交易日，对数轴）')
    ax[0].set_ylabel('累计超额收益 CAR（%）')
    ax[0].set_title(f'CAR 衰减曲线（N={len(detail)}）')
    ax[0].legend()
    ax[0].grid(alpha=.3)

    p = detail[f'pct{w[-1]}'].dropna() if w else pd.Series(dtype=float)
    if len(p):
        ax[1].hist(p, bins=np.arange(0, 101, 10), color='#7f8c8d', edgecolor='w')
        ax[1].axvline(50, color='gray', ls='--', lw=1, label='随机基准 50%')
        ax[1].set_xlabel(f'事件 CAR 在自身安慰剂分布中的分位（{w[-1]} 日，%）')
        ax[1].set_ylabel('事件数')
        ax[1].set_title('安慰剂分位分布')
        ax[1].legend()
        ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser(description='通用事件研究框架')
    ap.add_argument('--events', required=True, help='事件表 CSV（symbol,date[,label,note]）')
    ap.add_argument('--market', default='cn', choices=['cn', 'us'])
    ap.add_argument('--windows', default=','.join(map(str, DEFAULT_WINDOWS)))
    ap.add_argument('--out', default=f'{ROOT}/reports/event_study')
    ap.add_argument('--bench', default='ew', choices=['ew', 'csi'], help='A 股基准：ew 全市场等权 / csi 中证全指')
    ap.add_argument('--align', default='next', choices=['next', 'prev'])
    ap.add_argument('--lag', type=int, default=0, help='事件日后再延后几个交易日（盘后事件用 1）')
    ap.add_argument('--peers', default=None, help='美股同侪列表文件（每行一个 ticker）')
    a = ap.parse_args()

    ev = pd.read_csv(a.events, dtype={'symbol': str})
    ev['symbol'] = ev['symbol'].str.zfill(6) if a.market == 'cn' else ev['symbol'].str.upper()
    if 'label' not in ev.columns:
        ev['label'] = '未分类'
    ev['date'] = pd.to_datetime(ev['date'])
    windows = [int(x) for x in a.windows.split(',')]
    print(f'事件 {len(ev)} 条 | 市场 {a.market} | 窗口 {windows}', flush=True)

    mkt = CNMarket(bench=a.bench) if a.market == 'cn' else USMarket(peer_file=a.peers)

    rows = []
    for i, r in ev.iterrows():
        out = run_one(mkt, r['symbol'], r['date'], windows, align=a.align, lag=a.lag)
        out['label'] = r.get('label', '未分类')
        out['note'] = r.get('note', '')
        rows.append(out)
        if 'error' in out:
            print(f"  [{i + 1}/{len(ev)}] {r['symbol']} {r['date'].date()} 跳过：{out['error']}", flush=True)
        else:
            c = out.get(f'car{max(windows)}', np.nan)
            p = out.get(f'pct{max(windows)}', np.nan)
            print(f"  [{i + 1}/{len(ev)}] {out['symbol']} {out['date']} "
                  f"CAR{max(windows)} {c * 100:+.1f}% | 安慰剂分位 {p:.0f}%", flush=True)

    detail = pd.DataFrame(rows)
    detail = detail[~detail.get('error', pd.Series(dtype=str)).notna()] if 'error' in detail.columns else detail
    if len(detail) == 0:
        print('没有可用事件'); sys.exit(1)

    summ = summarize(detail, windows)
    detail.to_csv(f'{a.out}.csv', index=False)
    summ.to_csv(f'{a.out}_summary.csv', index=False)

    pw = max(windows)
    pc = [c for c in detail.columns if c.startswith('peer_')]
    peers_df = detail[['symbol', 'date', 'label'] + pc].dropna() if pc else None
    if peers_df is not None and len(peers_df):
        peers_df.to_csv(f'{a.out}_peers.csv', index=False)

    report(detail, summ, peers_df, mkt.name, windows, f'{a.out}.md')
    plot_car(detail, summ, windows, f'{a.out}.png')
    print(f'\n已输出：{a.out}.csv / _summary.csv / .md / .png')
    print('\n=== 汇总 ===')
    print(summ.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
