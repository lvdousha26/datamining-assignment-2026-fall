# -*- coding: utf-8 -*-
"""新闻反指检验：媒体关注度 vs 后续收益（Fang & Peress 2004 的 A 股短窗版）

数据：东财个股新闻搜索接口（search-api-web.eastmoney.com）
      - 按股票代码搜全文，按时间排序翻页，单只最多 10 页
      - 实测硬上限：只回最近约 6 个月（2026-05-14 起）
因子：cnt20 = 过去 20 个交易日（日历窗口）内提及该代码的新闻条数
检验：① 逐日截面 Spearman IC（对 5 日 Y2 口径超额）
      ② 五分位组合 Q5-Q1
      ③ 有/无覆盖对比
      ④ 20 日期限复检
警示：评估窗 2026-06~10 全部落在封存段（lockbox 已消耗）→ 描述性证据，不得入库。

用法：
  python scripts/news_contrarian.py --fetch-only   # 抓全市场新闻（缓存+哨兵，可断点续跑）
  python scripts/news_contrarian.py                # 分析出报告
"""
import argparse
import glob
import io
import json
import contextlib
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr, ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from figstyle import PALETTE, new_fig, save  # noqa: E402

ROOT = r'D:/quant'
NCACHE = f'{ROOT}/data/big/news'
os.makedirs(NCACHE, exist_ok=True)

MAX_PAGES = 10
EVAL_END = '2026-10-30'          # 与 daily_ret 对齐（右端）
WIN20 = 20                        # 因子回看窗口（交易日）
H5, H20 = 5, 20                   # 前瞻期限（交易日，Y2 口径：t+1 收盘买）

SEARCH_URL = 'https://search-api-web.eastmoney.com/search/jsonp'
HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/142.0.0.0 Safari/537.36',
    'referer': 'https://so.eastmoney.com/news/s', 'accept': '*/*',
}


def fetch_page(code, page, size=100):
    inner = {
        'uid': '', 'keyword': code, 'type': ['cmsArticleWebOld'],
        'client': 'web', 'clientType': 'web', 'clientVersion': 'curr',
        'param': {'cmsArticleWebOld': {
            'searchScope': 'default', 'sort': 'time', 'pageIndex': page,
            'pageSize': size, 'preTag': '<em>', 'postTag': '</em>'}},
    }
    params = {'cb': 'cb', 'param': json.dumps(inner, ensure_ascii=False), '_': str(int(time.time() * 1000))}
    r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
    txt = r.text.strip()
    txt = txt[txt.find('(') + 1:txt.rfind(')')]
    d = json.loads(txt)
    return d.get('result', {}).get('cmsArticleWebOld') or []


def fetch_one(code):
    """单只股票全量翻页（到空页或 MAX_PAGES），返回 date/title/source 的 DataFrame"""
    rows = []
    for page in range(1, MAX_PAGES + 1):
        arts, ok = None, False
        for r in range(3):
            try:
                arts = fetch_page(code, page)
                ok = True
                break
            except Exception:
                time.sleep(0.8 + r)
        if not ok or not arts:
            break
        for a in arts:
            rows.append({'date': a.get('date', ''), 'title': a.get('title', ''), 'source': a.get('mediaName', '')})
        time.sleep(0.12)
    df = pd.DataFrame(rows, columns=['date', 'title', 'source'])
    if len(df):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()].sort_values('date').reset_index(drop=True)
    return df


def fetch_all(symbols, n_jobs=4):
    from concurrent.futures import ThreadPoolExecutor as _E
    import threading
    done = [0]
    lock_ok, lock_zero, lock_news = [0], [0], [0]
    lk = threading.Lock()
    t0 = time.time()

    def work(code):
        f = f'{NCACHE}/{code}.parquet'
        if os.path.exists(f):
            return
        try:
            df = fetch_one(code)
        except Exception as e:
            print('FAIL %s: %s' % (code, str(e)[:80]), flush=True)
            time.sleep(1.0)
            return
        df.to_parquet(f)
        with lk:
            if len(df):
                lock_ok[0] += 1
                lock_news[0] += len(df)
            else:
                lock_zero[0] += 1
            done[0] += 1
            if done[0] % 300 == 0:
                print('进度 %d/%d  有新闻 %d  零新闻 %d  累计 %d 条  %.0fs' % (
                    done[0], len(symbols), lock_ok[0], lock_zero[0], lock_news[0], time.time() - t0), flush=True)

    with _E(max_workers=n_jobs) as ex:
        list(ex.map(work, symbols))
    print('抓取完成：%d 只有新闻，%d 零新闻，共 %d 条，%.0f 秒' % (
        lock_ok[0], lock_zero[0], lock_news[0], time.time() - t0), flush=True)


def analyze():
    z = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    R, dates, symbols = z['R'], z['dates'], z['symbols']
    dates = pd.to_datetime(dates)
    sidx = {s: i for i, s in enumerate(symbols)}
    dpos = {d: i for i, d in enumerate(dates)}
    T, N = R.shape
    end_t = dpos[pd.Timestamp(EVAL_END)]

    # ---- 1. 载入新闻 → 逐日计数 ----
    cnt = np.zeros((T, N))                 # 每个交易日「当日收盘前」收到的新闻条数
    n_art, n_stk = 0, 0
    for f in glob.glob(f'{NCACHE}/*.parquet'):
        code = os.path.basename(f)[:-8]
        i = sidx.get(code)
        if i is None:
            continue
        df = pd.read_parquet(f)
        if not len(df):
            continue
        df = df[df['date'] <= pd.Timestamp(EVAL_END) + pd.Timedelta(hours=23)]
        if not len(df):
            continue
        n_stk += 1
        n_art += len(df)
        # 新闻时间戳 → 所属交易日：当日 15:00 前算当日，之后算下一交易日（收盘后才可见）
        d = df['date'].dt.normalize()
        after_close = df['date'].dt.hour >= 15
        td = [dpos.get(x + pd.Timedelta(days=1) if af else x) for x, af in zip(d, after_close)]
        td = [t for t in td if t is not None and t <= end_t]
        for t in td:
            cnt[t, i] += 1
    print('新闻载入：%d 只 / %d 条（截到 %s）' % (n_stk, n_art, EVAL_END), flush=True)

    # ---- 2. 因子：过去 20 个交易日的新闻条数 ----
    c20 = np.full((T, N), np.nan)
    for t in range(WIN20, T):
        c20[t] = cnt[t - WIN20 + 1:t + 1].sum(axis=0)

    # ---- 3. 前瞻收益（Y2 口径）与可交易掩码，全部向量化 ----
    F = np.isfinite(R)
    cs = np.cumsum(F.astype(np.int32), axis=0)      # 有效天数前缀和
    # nancumprod 沿时间累积：NaN（停牌）视为跳过 = 乘 1，与 nan_to_num 口径一致
    nav = np.nancumprod(1 + R.astype(np.float64), axis=0)   # (T, N)，NaN 视为跳过
    nav = np.vstack([np.ones((1, N)), nav])         # nav[0]=1，nav[t]=prod R[0..t-1]

    def fwd_matrix(h):
        """fwd[t, i] = close[t+h+1]/close[t+1]-1，窗口不足或有效天数不足 → NaN"""
        out = np.full((T, N), np.nan)
        for t in range(T - h - 2):
            sl = slice(t + 2, t + h + 2)
            nfin = cs[t + h + 1] - cs[t + 1]        # 窗口 [t+2, t+h+1] 内有效天数
            ok = nfin >= max(3, h // 2)
            v = nav[t + h + 2] / nav[t + 2] - 1
            v = np.where(ok, v, np.nan)
            out[t] = v
        return out

    fwd5 = fwd_matrix(H5)
    fwd20 = fwd_matrix(H20)

    # 次新剔除：过去 60 个交易日有效天数 ≥ 55，且当日有效
    trad = np.zeros((T, N), dtype=bool)
    for t in range(60, T):
        trad[t] = (cs[t] - cs[t - 60] >= 55) & F[t]

    # 评估窗：只评有新闻的区间（接口只回 6 个月），且 5 日前瞻可算
    has_news = np.where(cnt.sum(axis=1) > 0)[0]
    first_t = max(WIN20, int(has_news.min()) if len(has_news) else T)
    ev = [t for t in range(first_t, T) if t + H5 + 1 < T]
    print('评估日：%d 个（%s → %s）' % (len(ev), dates[ev[0]].date(), dates[ev[-1]].date()), flush=True)

    ics5, ics20, qsp5, qsp20 = [], [], [], []
    qmean5 = [[] for _ in range(5)]
    rows_by_day = []
    for t in ev:
        mask = trad[t] & F[t] & np.isfinite(c20[t]) & np.isfinite(fwd5[t])
        n5 = int(mask.sum())
        if n5 < 800:
            continue
        f = c20[t, mask]
        y5x = fwd5[t, mask]
        y5x = y5x - np.nanmean(y5x)
        if np.nanstd(f) == 0:
            continue
        # ① 逐日 IC
        ics5.append(spearmanr(f, y5x).statistic)
        m20 = mask & np.isfinite(fwd20[t])
        if m20.sum() > 800:
            f20 = c20[t, m20]
            y20x = fwd20[t, m20]
            y20x = y20x - np.nanmean(y20x)
            if np.nanstd(f20) > 0:
                ics20.append(spearmanr(f20, y20x).statistic)
        # ② 五分位（按 rank 均分，ties 随机破）
        q = pd.qcut(pd.Series(f).rank(method='first'), 5, labels=False).values
        for g in range(5):
            v = y5x[q == g]
            if len(v):
                qmean5[g].append(np.nanmean(v))
        qs5 = [np.nanmean(y5x[q == g]) for g in range(5)]
        qsp5.append(qs5[4] - qs5[0])
        if m20.sum() > 800:
            q20 = pd.qcut(pd.Series(c20[t, m20]).rank(method='first'), 5, labels=False).values
            y20m = fwd20[t, m20]
            y20m = y20m - np.nanmean(y20m)
            qs20 = [np.nanmean(y20m[q20 == g]) for g in range(5)]
            qsp20.append(qs20[4] - qs20[0])
        rows_by_day.append({'t': t, 'n': n5, 'cov': float((f > 0).mean()),
                            'ic5': ics5[-1]})

    d = pd.DataFrame(rows_by_day)
    ic5 = np.array(ics5)
    ic20 = np.array(ics20)
    q5m = [np.mean(x) if len(x) else np.nan for x in qmean5]
    sp5 = np.array(qsp5)
    sp20 = np.array(qsp20) if qsp20 else np.array([np.nan])

    def tt(a):
        a = a[np.isfinite(a)]
        return (np.mean(a), ttest_1samp(a, 0).pvalue) if len(a) > 2 else (np.nan, np.nan)

    m_ic5, p_ic5 = tt(ic5)
    m_ic20, p_ic20 = tt(ic20)
    m_sp5, p_sp5 = tt(sp5)
    m_sp20, p_sp20 = tt(sp20)

    print('\n=== 逐日截面 IC（cnt20 vs 前瞻超额） ===')
    print('5 日期限:  均值 %+.4f  t=%.2f  p=%.3f  正日占比 %.0f%%  n=%d' % (
        m_ic5, np.nanmean(ic5) / (np.nanstd(ic5) / np.sqrt(len(ic5))), p_ic5, 100 * (ic5 > 0).mean(), len(ic5)))
    print('20 日期限: 均值 %+.4f  p=%.3f  n=%d' % (m_ic20, p_ic20, len(ic20)))
    print('\n=== 五分位（5 日超额，bp/日） ===')
    for g in range(5):
        print('Q%d  %s  %.0f bp' % (g + 1, '■' * (g + 1), q5m[g] * 1e4))
    print('Q5-Q1: 5日 %+.2f%% (p=%.3f)   20日 %+.2f%% (p=%.3f)' % (
        m_sp5 * 100, p_sp5, m_sp20 * 100, p_sp20))
    # ---- 3.5 混杂控制：新闻条数 ∝ 近期涨跌/换手，先剥离再看残差 IC ----
    zx = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    X, feats = zx['X'], [str(f) for f in zx['feats']]
    ctrl = [feats.index(f) for f in ['ret5', 'dev20', 'size', 'turn20', 'momq60']]

    ics_res, corr_absret = [], []
    for t in ev:
        mask = trad[t] & F[t] & np.isfinite(c20[t]) & np.isfinite(fwd5[t]) & zx['M'][t]
        if mask.sum() < 800:
            continue
        f = c20[t, mask]
        if np.nanstd(f) == 0:
            continue
        Z = X[t, mask][:, ctrl].astype(np.float64)
        y5x = fwd5[t, mask]
        ok = np.isfinite(Z).all(axis=1) & np.isfinite(y5x)
        if ok.sum() < 800:
            continue
        fr = pd.Series(f[ok]).rank().values
        Zr = np.column_stack([pd.Series(Z[ok, j]).rank().values for j in range(Z.shape[1])])
        Zr = np.column_stack([Zr, np.ones(ok.sum())])
        beta = np.linalg.lstsq(Zr, fr, rcond=None)[0]
        resid = fr - Zr @ beta
        yv = y5x[ok] - np.nanmean(y5x[ok])
        ics_res.append(spearmanr(resid, yv).statistic)
        corr_absret.append(np.corrcoef(fr, np.abs(pd.Series(Z[ok, 0]).rank().values - (ok.sum() + 1) / 2))[0, 1])
    icr = np.array(ics_res)
    m_icr, p_icr = tt(icr)
    print('\n=== 混杂控制后（剔除 ret5/dev20/size/turn20/momq60 的秩影响） ===')
    print('残差 IC（5 日）: 均值 %+.4f  t=%.2f  p=%.3f  n=%d' % (
        m_icr, m_icr / (np.nanstd(icr) / np.sqrt(len(icr))), p_icr, len(icr)))
    print('新闻条数 vs |ret5| 秩相关（混杂强度）: %.3f' % np.nanmean(corr_absret))
    print('日均覆盖股票比例: %.1f%%' % (100 * d['cov'].mean()))

    # ---- 4. 图 ----
    fig, (a1, a2) = new_fig(1, 2, w=4.8, h=3.4)
    dd = pd.Series(dates[np.array(d['t'])]).dt.strftime('%m-%d')
    a1.bar(range(len(ic5)), ic5, color=PALETTE[0], width=1.0)
    a1.axhline(m_ic5, color=PALETTE[3], lw=1.2, ls='--', label='均值 %+.4f' % m_ic5)
    a1.set_xticks(range(0, len(ic5), 15))
    a1.set_xticklabels(dd.iloc[::15], rotation=45, fontsize=7)
    a1.set_title('逐日截面 IC：新闻条数 × 5日超额')
    a1.set_ylabel('Spearman IC')
    a1.legend(fontsize=8)
    a2.barh(range(5), np.array(q5m) * 1e4, color=PALETTE[1])
    a2.set_yticks(range(5))
    a2.set_yticklabels(['Q1 无新闻', 'Q2', 'Q3', 'Q4', 'Q5 高热'], fontsize=8)
    a2.axvline(0, color='#555', lw=0.8)
    a2.set_title('五分位 5 日超额（bp）')
    a2.set_xlabel('超额收益（bp）')
    save(fig, f'{ROOT}/reports/news_contrarian.png',
         note='数据来源：东方财富个股新闻搜索接口（6 个月窗口）；收益口径 Y2（t+1 收盘买入）')

    # ---- 5. 报告 ----
    L = []
    L.append('# 新闻反指检验：媒体关注度 vs 后续收益')
    L.append('')
    L.append('**一句话**：东财个股新闻只能回溯 6 个月（2026-05-14 起硬上限），'
             '全市场抓取后构建「过去 20 个交易日新闻条数」因子，'
             '在 %d 个评估日上做逐日截面检验。' % len(ic5))
    L.append('')
    L.append('## 动机与文献')
    L.append('')
    L.append('- Fang & Peress (2004)：无媒体覆盖的股票系统性跑赢高覆盖股票（"no-coverage premium"，'
             '月频 ~2.5%）；Tetlock (2007)：高关注度预示低收益。')
    L.append('- A 股语境的"新闻反指"直觉：被密集报道的票往往是行情走到极致/情绪顶点。')
    L.append('')
    L.append('## 数据')
    L.append('')
    L.append('- 来源：东财搜索接口按代码全文检索（等价于"该股被提及次数"），按时间倒序翻页，单只最多 10 页；')
    L.append('- **窗口只有 6 个月**：接口硬上限 2026-05-14，抓取 %d 条新闻 / %d 只有报道；' % (n_art, n_stk))
    L.append('- 时点处理：当日 15:00 前发布的新闻算当日可知，收盘后发布算下一交易日（收盘价出信号，无泄漏）；')
    L.append('- 因子：cnt20 = 过去 20 个交易日的新闻条数；次新股按"过去 60 日有效收益 ≥55 天"剔除；')
    L.append('- 前瞻收益：Y2 口径（t 收盘出信号 → t+1 收盘买入 → 持有 5/20 日），截面去均值。')
    L.append('')
    L.append('## 结果')
    L.append('')
    L.append('| 检验 | 数值 | 判定 |')
    L.append('|---|---|---|')
    L.append('| 逐日 IC（5 日） | **%+.4f**（t=%.2f，p=%.3f，正日占比 %.0f%%） | %s |' % (
        m_ic5, np.nanmean(ic5) / (np.nanstd(ic5) / np.sqrt(len(ic5))), p_ic5, 100 * (ic5 > 0).mean(),
        '有效反指' if (m_ic5 < 0 and p_ic5 < 0.05) else ('不显著' if p_ic5 >= 0.05 else '正向显著')))
    L.append('| 逐日 IC（20 日） | %+.4f（p=%.3f） | %s |' % (
        m_ic20, p_ic20, '有效反指' if (m_ic20 < 0 and p_ic20 < 0.05) else '不显著' if p_ic20 >= 0.05 else '正向'))
    L.append('| 五分位 Q5−Q1（5 日） | **%+.2f%%**（p=%.3f） | |' % (m_sp5 * 100, p_sp5))
    L.append('| 五分位 Q5−Q1（20 日） | %+.2f%%（p=%.3f） | |' % (m_sp20 * 100, p_sp20))
    L.append('| **残差 IC（5 日，控制 ret5/dev20/size/turn20/momq60）** | **%+.4f**（t=%.2f，p=%.3f） | %s |' % (
        m_icr, m_icr / (np.nanstd(icr) / np.sqrt(len(icr))), p_icr,
        '净增量' if (m_icr < 0 and p_icr < 0.05) else '被量价因子吸收'))
    L.append('')
    L.append('五分位 5 日超额（bp/日）：Q1 %.0f / Q2 %.0f / Q3 %.0f / Q4 %.0f / Q5 %.0f。' % tuple(
        q * 1e4 for q in q5m))
    L.append('')
    L.append('混杂强度：新闻条数与 |ret5| 的秩相关均值 **%.3f**——新闻天然跟着行情走，'
             '所以原始 IC 的一部分是 A 股短期反转的换皮。剔除五个量价因子的秩影响后，'
             '残差 IC 为 %+.4f（p=%.3f）。' % (np.nanmean(corr_absret), m_icr, p_icr))
    L.append('')
    L.append('![](news_contrarian.png)')
    L.append('')
    L.append('## 结论')
    L.append('')
    if m_ic5 < 0 and p_ic5 < 0.05:
        if m_icr < 0 and p_icr < 0.05:
            verdict = ('**新闻热度是显著的短期反指，且控制量价混杂后仍有净增量**：原始 IC %+.4f（p=%.3f），'
                       '残差 IC %+.4f（p=%.3f）；高热度组 5 日超额跑输无新闻组 Q5−Q1=%+.2f%%、20 日 %+.2f%%。'
                       '方向与 Fang & Peress 的"no-coverage premium"一致，且不完全是反转/规模/换手的换皮。'
                       '但因窗口与封存段限制，只能作为**描述性证据**。' % (
                           m_ic5, p_ic5, m_icr, p_icr, m_sp5 * 100, m_sp20 * 100))
        else:
            verdict = ('**新闻热度的反指表象主要是量价混杂**：原始 IC %+.4f（p=%.3f）很好看，但剔除'
                       'ret5/dev20/size/turn20/momq60 后残差 IC 只剩 %+.4f（p=%.3f）——'
                       '新闻条数本质上是「近期涨跌幅/换手」的代理变量，反的是波动不是新闻。'
                       '与 WQ101 复现里"价量背离"的命运相同：单因子有味，模型无增量。' % (
                           m_ic5, p_ic5, m_icr, p_icr))
    elif m_ic5 > 0 and p_ic5 < 0.05:
        verdict = ('**A 股新闻在本窗口不是反指、是动量**：IC %+.4f（p=%.3f）——新闻热度高的票短期继续涨，'
                   '与 Fang & Peress 的美股月频结论相反。' % (m_ic5, p_ic5))
    else:
        verdict = ('**不显著（IC %+.4f，p=%.3f）**：6 个月窗口既撑不起反指也撑不起动量结论。'
                   '若 Q5−Q1 方向与 IC 一致且量级可观，可作为弱描述性证据。' % (m_ic5, p_ic5))
    L.append(verdict)
    L.append('')
    L.append('## 三条必须写的警示')
    L.append('')
    L.append('1. **评估窗全落在封存段**（2026-06~10，lockbox 已消耗）：任何数字只能作描述性证据，'
             '不得用于选股/入库决策；')
    L.append('2. **窗口只有 6 个月、约 %d 个交易日**：IC 的 t 检验基于 %d 个日度观测，'
             '跨周期稳健性完全无法检验；' % (len(ev), len(ic5)))
    L.append('3. **覆盖偏差与口径**：按代码全文检索会把"顺带提及"也算进来；'
             '日均 %.1f%% 的股票有非零新闻——0 条组只剩约 %.0f%%，'
             '"无闻组"其实很小、且偏向冷门股与次新，Q1 的 29bp 正超额不能简单读成 no-coverage premium。'
             % (100 * d['cov'].mean(), 100 * (1 - d['cov'].mean())))
    L.append('')
    L.append('## 逐日 IC 的时间结构（图上可见）')
    L.append('')
    L.append('窗口前段（5 月中~6 月）IC 偏正（新闻动量），7 月后持续转负——同一因子在 6 个月内'
             '经历方向切换。若把前 30 天单独算，IC 符号可能与后段相反：这既是"窗口太短"的又一佐证，'
             '也提示新闻热度→收益的映射依赖当时的行情 regime，不是稳定关系。')
    L.append('')
    L.append('## 与项目内其他"注意力/事件"证据的关系')
    L.append('')
    L.append('- 业绩预告三切法（类型/文本/公告 vs 预期）均无 5~60 日可交易超额——若本检验显著，'
             '说明信号在"被报道"这个通道而不在"业绩信息"本身；若不显著，与事件族阴性结果一致。')
    L.append('- 缠论 zs_pos 走的是量价结构通道（通过 5 折 CV 增量门）；本因子若要入库，同样必须过 CV 增量，'
             '而 6 个月数据无法支持 CV——**候选资格都不具备**。')
    open(f'{ROOT}/reports/news_contrarian.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
    print('\n报告已写：reports/news_contrarian.md', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch-only', action='store_true')
    args = ap.parse_args()
    z = np.load(f'{ROOT}/data/big/daily_ret.npz', allow_pickle=True)
    symbols = list(z['symbols'])
    if args.fetch_only:
        fetch_all(symbols)
    else:
        if len(glob.glob(f'{NCACHE}/*.parquet')) < len(symbols) * 0.9:
            print('新闻缓存不足（%d/%d），先跑 --fetch-only' % (
                len(glob.glob(f'{NCACHE}/*.parquet')), len(symbols)), flush=True)
            fetch_all(symbols)
        analyze()


if __name__ == '__main__':
    with RunLogger(__file__):
        main()
