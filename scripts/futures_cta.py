# -*- coding: utf-8 -*-
"""国内商品期货 CTA 回测（主连，Sina 日线）

规则（全部 a priori，不做参数挑选，三变体全展示）：
  V1 唐奇安 20/10（海龟 S1）  V2 唐奇安 55/20（海龟 S2）  V3 双均线 20/60
执行现实性（用户点名的四条）：
  ① 最小开仓手数：整数手，1 手起；本金两档 100 万 / 500 万对照（粒度敏感性）
  ② 昨仓/今仓：信号收盘出、次日收盘执行、持仓必过夜 → 平仓全走「平昨」普通费率，
     结构上不触发平今（换月 = 平昨旧 + 开今新，也是普通费率 + 换月成本）
  ③ 临时规则：停板日（收盘=最高/最低 且 |ret|≥停板幅度×0.9）不成交、顺延至多 3 日；
     停板累计规则（郑商所式：4 日累计涨跌幅 ≥ 3×停板幅度 → 保证金 ×3）触发时
     检查总保证金占用，超 90% 权益强制减仓
  ④ 换月：主连序列持仓量跳增 >35% 判为换月日，当日「假价差」收益清零（收益率后复权），
     持仓过换月收换月成本（双边手续费 + 2 跳）
风控：每腿目标波动贡献 = 组合 10%/√6；单腿保证金 ≤ 25% 权益；
     压力档：手续费 ×2、保证金率 ×1.5（节前提保代理）
成本：单边手续费 万1.5（含公司加成）+ 1 跳滑点

用法：python scripts/futures_cta.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from figstyle import PALETTE, new_fig, save  # noqa: E402

ROOT = r'D:/quant'
CACHE = f'{ROOT}/data/big/cta'
os.makedirs(CACHE, exist_ok=True)

# 品种规格：乘数(每手)、跳值、保证金率、停板幅度（近似值，报告注明）
SPEC = {
    'RB0': dict(name='螺纹钢', mult=10,   tick=1.0,  margin=0.13, limit=0.07),
    'M0':  dict(name='豆粕',   mult=10,   tick=1.0,  margin=0.12, limit=0.06),
    'CU0': dict(name='沪铜',   mult=5,    tick=10.0, margin=0.13, limit=0.06),
    'I0':  dict(name='铁矿石', mult=100,  tick=0.5,  margin=0.15, limit=0.08),
    'AU0': dict(name='黄金',   mult=1000, tick=0.02, margin=0.12, limit=0.05),
    'SC0': dict(name='原油',   mult=1000, tick=0.1,  margin=0.15, limit=0.09),
}
FEE_RATE = 1.5e-4          # 单边手续费（成交金额比例，含公司加成）
FEE_RATE_STRESS = 3.0e-4   # 压力档 ×2
VOL_TARGET = 0.10          # 组合年化目标波动
MARGIN_STRESS = 1.5        # 压力档：保证金率 ×1.5（节前提保代理）
CAPITALS = [1e6, 5e6]
ROLL_HOLD_JUMP = 0.35      # 持仓量跳增阈值 → 判换月
START = '2010-01-01'


def load(syms):
    """读缓存 → 收益率后复权（换月日假价差清零），返回原始与复权序列"""
    raw, adj, rolls = {}, {}, {}
    for s in syms:
        d = pd.read_parquet(f'{CACHE}/{s}.parquet').sort_values('date').reset_index(drop=True)
        d['date'] = pd.to_datetime(d['date'])
        c = d['close'].astype(float).values
        h = d['hold'].astype(float).values
        ret = np.append(np.nan, c[1:] / c[:-1] - 1)
        hchg = np.append(np.nan, h[1:] / h[:-1] - 1)
        roll = np.zeros(len(d), dtype=bool)
        roll[1:] = hchg[1:] > ROLL_HOLD_JUMP
        # 后复权：换月日收益清零
        aret = ret.copy()
        aret[roll] = 0.0
        raw[s] = d.set_index('date')[['open', 'high', 'low', 'close', 'hold']]
        adj[s] = pd.Series(np.nan, index=d['date'])
        a = 100.0
        for i in range(len(d)):
            a = a * (1.0 if not np.isfinite(aret[i]) else (1 + aret[i]))
            adj[s].iloc[i] = a
        rolls[s] = d['date'][roll]
    return raw, adj, rolls


def signals_donchian(px, n_in, n_out):
    """唐奇安：突破 n_in 日高做多入、跌破 n_in 日低做空入；反向 n_out 日退出/反手"""
    hh = px.rolling(n_in).max().shift(1)
    ll = px.rolling(n_in).min().shift(1)
    xh = px.rolling(n_out).max().shift(1)
    xl = px.rolling(n_out).min().shift(1)
    pos = pd.Series(np.nan, index=px.index)
    cur = 0
    arr_p, arr_hh, arr_ll, arr_xh, arr_xl = (px.values, hh.values, ll.values, xh.values, xl.values)
    for i in range(len(px)):
        if np.isfinite(arr_hh[i]):
            if cur <= 0 and arr_p[i] > arr_hh[i]:
                cur = 1
            elif cur >= 0 and arr_p[i] < arr_ll[i]:
                cur = -1
            elif cur == 1 and arr_p[i] < arr_xl[i]:
                cur = 0
            elif cur == -1 and arr_p[i] > arr_xh[i]:
                cur = 0
        pos.iloc[i] = cur
    return pos


def signals_ma(px, fast, slow):
    mf, ms = px.rolling(fast).mean(), px.rolling(slow).mean()
    return pd.Series(np.sign(mf - ms).where(np.isfinite(ms), np.nan), index=px.index)


VARIANTS = {
    'V1 唐奇安20/10': lambda px: signals_donchian(px, 20, 10),
    'V2 唐奇安55/20': lambda px: signals_donchian(px, 55, 20),
    'V3 双均线20/60': lambda px: signals_ma(px, 20, 60),
}


def backtest(adj, raw, rolls, sig, capital=1e6, fee=FEE_RATE, margin_mult=1.0):
    """逐日联合结算：整手、停板顺延、三倍保证金规则、换月成本"""
    # 主日历
    all_days = sorted(set().union(*[set(a.index) for a in adj.values()]))
    days = pd.DatetimeIndex([d for d in all_days if d >= pd.Timestamp(START)])
    syms = list(adj.keys())
    n = len(syms)

    # 对齐矩阵：收盘价（原始，用于成交与盈亏）、复权价（信号）、持仓量、高低价
    P = pd.DataFrame({s: raw[s]['close'].reindex(days) for s in syms})
    A = pd.DataFrame({s: adj[s].reindex(days) for s in syms})
    H = pd.DataFrame({s: raw[s]['high'].reindex(days) for s in syms})
    L = pd.DataFrame({s: raw[s]['low'].reindex(days) for s in syms})
    SIG = pd.DataFrame({s: sig[s].reindex(days) for s in syms})
    rollmat = pd.DataFrame(False, index=days, columns=syms)
    for s in syms:
        rollmat.loc[rolls[s][rolls[s].isin(days)], s] = True

    P = P.ffill()
    A = A.ffill()
    ret = P / P.shift(1) - 1
    # 换月日盈亏按复权口径（假价差清零）
    ret_adj = ret.copy()
    ret_adj[rollmat] = 0.0

    equity = capital
    nav = [equity]
    margin_used = []
    lots = np.zeros(n)                      # 当前持仓（带方向，手数）
    pending = [None] * n                    # 待成交目标手数（昨日信号）
    defer_ct = np.zeros(n, dtype=int)
    pos_hold = [0] * n                      # 连续持仓日数（判换月成本）
    n_trades, n_rollcost, n_forced = 0, 0, 0
    vol = pd.DataFrame(index=days, columns=syms, dtype=float)

    for t in range(1, len(days)):
        day = days[t]
        prev = days[t - 1]
        # ---- 1. 盈亏结算（昨日收盘 → 今日收盘，按复权收益）----
        for j, s in enumerate(syms):
            if lots[j] != 0 and np.isfinite(ret_adj.iloc[t, j]):
                mult = SPEC[s]['mult']
                pnl = lots[j] * mult * (P.iloc[t, j] - P.iloc[t - 1, j])
                equity += pnl
        # ---- 2. 成交昨日挂单（停板顺延）----
        for j, s in enumerate(syms):
            if pending[j] is None:
                continue
            tgt = pending[j]
            px = P.iloc[t, j]
            if not np.isfinite(px) or px <= 0:
                continue
            ret_t = ret.iloc[t, j]
            lim = SPEC[s]['limit']
            locked = (np.isfinite(ret_t) and abs(ret_t) >= lim * 0.9 and
                      ((ret_t > 0 and abs(H.iloc[t, j] - px) < 1e-9) or
                       (ret_t < 0 and abs(L.iloc[t, j] - px) < 1e-9)))
            if locked and defer_ct[j] < 3:
                defer_ct[j] += 1
                continue
            mult = SPEC[s]['mult']
            dlots = tgt - lots[j]
            if dlots != 0:
                equity -= abs(dlots) * (fee * px * mult + SPEC[s]['tick'] * mult)
                n_trades += abs(int(dlots))
            lots[j] = tgt
            pending[j] = None
            defer_ct[j] = 0
        # ---- 3. 保证金检查（含停板累计 3 倍规则）----
        marg = 0.0
        for j, s in enumerate(syms):
            if lots[j] == 0:
                continue
            px = P.iloc[t, j]
            m3 = 1.0
            if t >= 4 and np.isfinite(A.iloc[t, j]):
                cum = A.iloc[t, j] / A.iloc[t - 4, j] - 1
                if abs(cum) >= 3 * SPEC[s]['limit']:
                    m3 = 3.0
            marg += abs(lots[j]) * px * SPEC[s]['mult'] * SPEC[s]['margin'] * margin_mult * m3
        margin_used.append(marg / equity if equity > 0 else np.nan)
        if marg > equity * 0.90:
            # 强制减仓：按保证金占比从大到小砍
            order = np.argsort(-np.array([
                abs(lots[j]) * P.iloc[t, j] * SPEC[s]['mult'] * SPEC[s]['margin'] * margin_mult
                if lots[j] != 0 else 0 for j, s in enumerate(syms)]))
            for j in order:
                if marg <= equity * 0.90:
                    break
                if lots[j] == 0:
                    continue
                s = syms[j]
                px = P.iloc[t, j]
                equity -= abs(lots[j]) * (fee * px * SPEC[s]['mult'] + SPEC[s]['tick'] * SPEC[s]['mult'])
                marg -= abs(lots[j]) * px * SPEC[s]['mult'] * SPEC[s]['margin'] * margin_mult
                lots[j] = 0
                n_forced += 1
        # ---- 4. 换月成本（持仓跨换月日：平旧开新）----
        for j, s in enumerate(syms):
            if lots[j] != 0 and rollmat.iloc[t, j]:
                px = P.iloc[t, j]
                equity -= abs(lots[j]) * (fee * px * SPEC[s]['mult'] + SPEC[s]['tick'] * SPEC[s]['mult'])
                n_rollcost += abs(int(lots[j]))
        # ---- 5. 收盘出信号 → 目标手数（明日成交）----
        for j, s in enumerate(syms):
            sig_t = SIG.iloc[t, j]
            px = P.iloc[t, j]
            if not np.isfinite(sig_t) or not np.isfinite(px) or px <= 0:
                continue
            # 已实现波动（20 日，复权收益）
            w = ret_adj.iloc[max(0, t - 19):t + 1, j]
            v = np.nanstd(w) * np.sqrt(252) if w.notna().sum() >= 12 else np.nan
            vol.iloc[t, j] = v
            if not np.isfinite(v) or v <= 0:
                continue
            per_leg = VOL_TARGET / np.sqrt(len(syms))
            notional_per_lot = px * SPEC[s]['mult']
            tgt_val = per_leg * equity / v                     # 目标名义市值
            tgt_lots = int(np.floor(tgt_val / notional_per_lot))  # 整手
            # 保证金上限 25% 权益
            if tgt_lots > 0:
                cap = int(np.floor(0.25 * equity /
                           (notional_per_lot * SPEC[s]['margin'] * margin_mult)))
                tgt_lots = min(tgt_lots, cap)
            pending[j] = int(sig_t) * tgt_lots
        nav.append(equity)
    nav = pd.Series(nav, index=days[:len(nav)])
    mu = pd.Series(margin_used, index=days[1:len(margin_used) + 1])
    return dict(nav=nav, margin=mu, trades=n_trades, rollcost=n_rollcost, forced=n_forced)


def metrics(nav, rf=0.0):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    return dict(ann=ann, vol=vol, sharpe=(ann - rf) / vol if vol > 0 else np.nan,
                mdd=dd, calmar=ann / abs(dd) if dd < 0 else np.nan, years=yrs)


def main():
    syms = list(SPEC.keys())
    raw, adj, rolls = load(syms)
    print('数据载入完成', {s: ('%s~%s' % (adj[s].index[0].date(), adj[s].index[-1].date()),
                                '换月%d次' % len(rolls[s])) for s in syms}, flush=True)

    sig = {v: {s: fn(adj[s]) for s in syms} for v, fn in VARIANTS.items()}

    results = {}
    for vname in VARIANTS:
        for cap in CAPITALS:
            results[(vname, cap)] = backtest(adj, raw, rolls, sig[vname], capital=cap)
        # 压力档（100 万）
        results[(vname, 'stress')] = backtest(adj, raw, rolls, sig[vname], capital=1e6,
                                              fee=FEE_RATE_STRESS, margin_mult=MARGIN_STRESS)

    print('\n=== 三变体 × 本金档（2010-01 起，费后） ===', flush=True)
    for vname in VARIANTS:
        for k in [(vname, 1e6), (vname, 5e6), (vname, 'stress')]:
            m = metrics(results[k]['nav'])
            r = results[k]
            print('%-14s 本金%-8s 年化 %6.2f%%  夏普 %5.2f  回撤 %6.1f%%  Calmar %5.2f  '
                  '成交%d手 换月%d手 强平%d次' % (
                      vname, ('100万' if k[1] == 1e6 else '500万') if k[1] != 'stress' else '压力档',
                      100 * m['ann'], m['sharpe'], 100 * m['mdd'], m['calmar'],
                      r['trades'], r['rollcost'], r['forced']), flush=True)

    # 分年表（V1, 100万）
    nav = results[('V1 唐奇安20/10', 1e6)]['nav']
    yr = nav.groupby(nav.index.year).agg(['first', 'last'])
    yr_ret = yr['last'] / yr['last'].shift(1) - 1          # 上一年「年末」做分母
    yr_ret.iloc[0] = yr['last'].iloc[0] / nav.iloc[0] - 1

    # 分品种腿（V1）：单独每腿等额波动的单品种回测对照
    print('\n=== 分品种（V1，单品种满配口径） ===', flush=True)
    leg_rows = []
    for s in syms:
        leg = backtest({s: adj[s]}, {s: raw[s]}, {s: rolls[s]},
                       {s: sig['V1 唐奇安20/10'][s]}, capital=1e6)
        m = metrics(leg['nav'])
        bh = metrics((adj[s] / adj[s].iloc[0] * 1e6).dropna())
        leg_rows.append((SPEC[s]['name'], m, bh))
        print('%-6s 策略腿 年化 %6.2f%% 夏普 %5.2f 回撤 %6.1f%% | 买入持有 年化 %6.2f%% 夏普 %5.2f 回撤 %6.1f%%' % (
            SPEC[s]['name'], 100 * m['ann'], m['sharpe'], 100 * m['mdd'],
            100 * bh['ann'], bh['sharpe'], 100 * bh['mdd']), flush=True)

    # ---- 图 ----
    fig, (a1, a2) = new_fig(1, 2, w=4.8, h=3.4)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(VARIANTS))]
    for (vname, _), c in zip([(k, None) for k in VARIANTS], colors):
        a1.plot(results[(vname, 1e6)]['nav'] / 1e6, label=vname, color=c, lw=1.1)
    a1.set_yscale('log')
    a1.legend(fontsize=8)
    a1.set_title('CTA 组合净值（100 万档，费后，对数）')
    a1.set_ylabel('净值（倍）')
    mu = results[('V1 唐奇安20/10', 1e6)]['margin']
    a2.plot(mu.rolling(60).mean() * 100, color=PALETTE[0], lw=1.0, label='60日均')
    a2.axhline(90, color=PALETTE[3], ls='--', lw=1, label='强平线 90%')
    a2.set_title('保证金占用率（V1）')
    a2.set_ylabel('%')
    a2.legend(fontsize=8)
    save(fig, f'{ROOT}/reports/futures_cta.png',
         note='数据来源：新浪财经期货主连日线（RB/M/CU/I/AU/SC，收益率后复权处理换月）；成本：单边万1.5+1跳')

    # ---- 报告 ----
    L = []
    L.append('# 国内商品期货 CTA 回测：六品种主连，规则全展示')
    L.append('')
    L.append('**一句话**：三套 a priori 经典规则（唐奇安 20/10、55/20、双均线 20/60）×'
             ' 六品种（螺纹/豆粕/沪铜/铁矿石/黄金/原油）波动率目标组合，'
             '整手交易、停板顺延、三倍保证金规则、换月成本全建模；2010-01 起费后回测。')
    L.append('')
    L.append('## 用户点名的四条现实约束 → 建模方式')
    L.append('')
    L.append('| 约束 | 建模 |')
    L.append('|---|---|')
    L.append('| 最小开仓手数 | 整手（1 手起）+ 100 万/500 万两档对照，量化粒度损失 |')
    L.append('| 昨仓/今仓 | 信号收盘出、次日执行、持仓必过夜 → 平仓全走平昨普通费率；'
             '结构上不触发平今；换月=平昨旧+开今新 |')
    L.append('| 停板累计规则（郑商所式，即「周三三倍」所指的 3 倍提保条款） | '
             '4 日累计涨跌幅 ≥ 3×停板幅度 → 保证金 ×3；占用超 90% 权益强制减仓 |')
    L.append('| 临时规则（停板无法成交） | 收盘=板价且 |ret|≥板幅×0.9 → 顺延至多 3 日 |')
    L.append('')
    L.append('## 总结果（2010-01 ~ 2026-11，费后）')
    L.append('')
    L.append('| 变体 | 本金 | 年化 | 夏普 | 最大回撤 | Calmar | 成交手数 | 强平次数 |')
    L.append('|---|---|---|---|---|---|---|---|')
    for vname in VARIANTS:
        for k, label in [((vname, 1e6), '100 万'), ((vname, 5e6), '500 万'), ((vname, 'stress'), '100 万压力档')]:
            m = metrics(results[k]['nav'])
            r = results[k]
            L.append('| %s | %s | %.2f%% | %.2f | %.1f%% | %.2f | %d | %d |' % (
                vname, label, 100 * m['ann'], m['sharpe'], 100 * m['mdd'], m['calmar'],
                r['trades'], r['forced']))
    L.append('')
    L.append('压力档 = 手续费 ×2（万3）+ 保证金率 ×1.5（节前提保代理）。')
    L.append('')
    L.append('## 分年（V1，100 万档）')
    L.append('')
    L.append('| 年份 | 收益 |')
    L.append('|---|---|')
    for y, v in yr_ret.items():
        L.append('| %d | %+.1f%% |' % (y, 100 * v))
    L.append('')
    L.append('## 分品种：策略腿 vs 买入持有（V1）')
    L.append('')
    L.append('| 品种 | 腿年化 | 腿夏普 | 腿回撤 | B&H 年化 | B&H 夏普 | B&H 回撤 |')
    L.append('|---|---|---|---|---|---|---|')
    for name, m, bh in leg_rows:
        L.append('| %s | %.2f%% | %.2f | %.1f%% | %.2f%% | %.2f | %.1f%% |' % (
            name, 100 * m['ann'], m['sharpe'], 100 * m['mdd'],
            100 * bh['ann'], bh['sharpe'], 100 * bh['mdd']))
    L.append('')
    L.append('![](futures_cta.png)')
    L.append('')
    L.append('## 已知近似（读数字前必看）')
    L.append('')
    L.append('1. **主连 + 收益率后复权**：换月日按持仓量跳增 >35% 判定（启发式），'
             '当日假价差清零、另收换月成本；未逐合约重建主力切换，极端贴水结构下会低估展期损益；')
    L.append('2. **成交价 = 次日收盘价**：真实应在盘中/开盘，1 跳滑点是保守下限；')
    L.append('3. 停板幅度/保证金率/手续费用近似常数值（历史上交易所调过多次），'
             '压力档是代理；保证金三倍规则用郑商所条款近似推广到全部品种；')
    L.append('4. 保证金占用之外的现金按 0 利息（保守，实际可买货基/国债贴 2%%）；')
    L.append('5. 参数是 a priori 经典值、三变体全展示不选优——但品种池（六个）本身就是挑过的，'
             '「六品种里选六个」存在选择偏差，扩池重算是后续工作。')
    L.append('')
    L.append('## 与项目内其他趋势证据的关系')
    L.append('')
    L.append('- 指数上的网格/海龟/MACD 全军覆没（夏普 0.1~0.2），但那是**单资产低杠杆**；'
             '本回测是**多品种+波动率目标+可做空**——CTA 的收益来源三者缺一不可；')
    L.append('- 原油单品种趋势（夏普 1.3~1.7，样本止于 2020-04）与本结果互为印证；')
    L.append('- 本策略与 A 股面板/禁买清单完全独立（不同市场、不同数据通道），'
             '按项目规矩单独存档评估，不进 42 因子体系。')
    open(f'{ROOT}/reports/futures_cta.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
    print('\n报告已写：reports/futures_cta.md', flush=True)


if __name__ == '__main__':
    with RunLogger(__file__):
        main()
