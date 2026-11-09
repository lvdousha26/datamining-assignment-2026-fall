# -*- coding: utf-8 -*-
"""降频 + 缓冲带：换手-超额收益前沿（只在开发段扫描，不碰 lockbox）

调仓周期 step ∈ {5,10,20,40} 交易日 × 缓冲带 buffer ∈ {0,.1,.2,.4,.8}
两种组合：① top10% 纯多头 ② 剔除垫底 10%（负向筛选）
持有 step 日的收益 = step/5 段非重叠 5 日收益之和（Y2 已按 5 日切分）
成本：单边 20bp，换手 T 对应买卖各 T，合计 T×0.004
"""
import numpy as np, pandas as pd

import sys
ROOT = r'D:/quant'
V1 = '--v1' in sys.argv          # 用 v1 备份做方法验证，不干扰 v2 重训
SUB = '/v1' if V1 else ''
z = np.load(f'{ROOT}/data/big{SUB}/arr.npz', allow_pickle=True)
Y2, M = z['Y2'], z['M']
p = np.load(f'{ROOT}/data/big{SUB}/preds.npz')
tt, ii, pred = p['tt_te'], p['ii_te'], p['p_st_te']
d = pd.DataFrame({'t': tt, 'i': ii, 'p': pred, 'y': Y2[tt, ii]}).dropna()
all_t = np.sort(d['t'].unique())
GROUPS = {t: g.sort_values('p', ascending=False).reset_index(drop=True)
          for t, g in d.groupby('t')}
mkt = {t: g['y'].mean() for t, g in GROUPS.items()}
Q, COST = .1, .004
print('开发段 %d 行 / %d 日 | %s ~ %s' %
      (len(d), len(all_t), pd.to_datetime(z['dates'][all_t[0]]).date(),
       pd.to_datetime(z['dates'][all_t[-1]]).date()), flush=True)

def run(step, buf, mode):
    reb = all_t[::step][:-1]
    rec, turns, prev = [], [], set()
    for t0 in reb:
        g = GROUPS.get(t0)
        if g is None or len(g) < 100: continue
        n = max(int(len(g) * Q), 10)
        pct = 1 - np.arange(len(g)) / len(g)          # 得分百分位，1 为最高
        rank = dict(zip(g['i'].values, pct))
        if mode == 'top':
            keep = [i for i in prev if rank.get(i, 0) >= 1 - Q - buf] if prev else []
            cand = [i for i in g['i'].values if i not in set(keep)]
            hold = (keep + cand[:max(n - len(keep), 0)])[:n]
            bench_i = np.array(hold, dtype=int)
        else:
            # 缓冲带：上期被剔除的，只要排名仍在 (Q+buf) 垫底区内就继续剔除，
            # 优先保留 → buffer 越大名单越稳；剩余名额由当前最低分补足。
            keep_bot = {i for i in prev if rank.get(i, 0) <= Q + buf} if prev else set()
            bot = set(list(keep_bot)[:n])
            need = n - len(bot)
            if need > 0:
                bot |= set([i for i in g['i'].values[-n:] if i not in bot][:need])
            rest_i = g.loc[~g['i'].isin(bot), 'i'].values.astype(int)
            hold = bot
            bench_i = rest_i
        cur = set(hold)
        turn = len(cur - prev) / max(len(cur), 1) if prev else 1.0
        turns.append(turn); prev = cur
        nseg = step // 5
        rs = [Y2[min(t0 + 5 * k, len(Y2) - 1), bench_i] for k in range(nseg)]
        r = np.nanmean(np.nansum(np.vstack(rs), 0))
        bench = sum(mkt.get(t0 + 5 * k, 0) for k in range(nseg))
        rec.append((r - bench, turn))
    if not rec: return None
    ex = np.array([a for a, _ in rec]); tn = np.array([b for _, b in rec])
    wt = Q if mode == 'exbot' else 1.0
    net = ex - tn * COST * wt                       # 成本按组合权重换手计
    periods = 252.0 / step                          # 每年调仓次数，随周期变化
    ann = (1 + net).prod() ** (periods / len(net)) - 1
    vol = net.std() * np.sqrt(periods)
    return dict(step=step, buffer=buf, mode=mode,
                turnover=tn.mean() * wt,                    # 每次调仓的组合权重换手
                ann_turnover=tn.mean() * wt * periods,      # 年化换手（调仓次数随周期下降）
                list_turnover=tn.mean(),
                gross=ex.mean(), net=net.mean(), ann=ann, vol=vol,
                sharpe=(ann - .02) / vol if vol > 0 else np.nan, n=len(net))

rows = []
for mode in ['top', 'exbot']:
    for step in [5, 10, 20, 40]:
        for buf in [0, .1, .2, .4, .8]:
            r = run(step, buf, mode)
            if r: rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(f'{ROOT}/reports/turnover_frontier.csv', index=False)

print('\n=== 剔除垫底 10%（负向筛选）：换手-收益前沿 ===')
e = R[R['mode'] == 'exbot'].copy()
print('%-6s %-7s %9s %11s %11s %8s' % ('周期', 'buffer', '换手/期', '年化换手', '净超额/期', '夏普'))
for _, r in e.iterrows():
    print('%-6d %-7.1f %8.1f%% %10.0f%% %10.4f%% %8.2f' %
          (r.step, r.buffer, r.turnover * 100, r.ann_turnover * 100, r.net * 100, r.sharpe))

print('\n=== top10% 纯多头 ===')
t = R[R['mode'] == 'top'].copy()
for _, r in t.iterrows():
    print('%-6d %-7.1f %8.1f%% %10.0f%% %10.4f%% %8.2f' %
          (r.step, r.buffer, r.turnover * 100, r.ann_turnover * 100, r.net * 100, r.sharpe))

for mode, label in [('exbot', '负向筛选'), ('top', '纯多头')]:
    s = R[(R['mode'] == mode) & (R.turnover < .10)]
    if len(s):
        b = s.loc[s.sharpe.idxmax()]
        print('\n%s：换手<10%% 的最优配置 = 周期 %d 日 / buffer %.1f → 换手 %.1f%% | 净超额 %.2f%%/年 | 夏普 %.2f' %
              (label, b.step, b.buffer, b.turnover * 100, b.ann * 100, b.sharpe))
    else:
        print('\n%s：没有任何配置把换手压到 10%% 以下（最低 %.1f%%）' %
              (label, R[R['mode'] == mode].turnover.min() * 100))
print('\n已存 reports/turnover_frontier.csv')
