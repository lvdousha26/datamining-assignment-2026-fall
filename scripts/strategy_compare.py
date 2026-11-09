# -*- coding: utf-8 -*-
"""策略对照：全市场剔底 vs 行业内剔底（同段/同预测/同成本）

段：2024-01 ~ 2025-06（preds.npz 的 me + te，LGBM 在 <2024 训练，两段都是 OOS）
预测：p_lgb（LGBM 单模型）—— 不使用 p_st（Stacking 在 me 段是样本内）
调仓：每 5 个交易日，5 日持有（Y2 口径）
成本：单边 15 / 20 bp，按持仓权重更替率
基线：全市场等权（Y2 已截面去均值，故组合 Y2 均值即超额）
"""
import numpy as np, pandas as pd

ROOT = 'D:/quant'
STEP, PPY = 5, 252 / 5

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
Y2, M = z['Y2'], z['M']
dates = pd.to_datetime(z['dates'])
IND_sw = np.load(f'{ROOT}/data/big/IND_sw.npy')
IND_cl = np.load(f'{ROOT}/data/big/IND_cluster.npy')
T, N = Y2.shape

p = np.load(f'{ROOT}/data/big/preds.npz')
tt = np.concatenate([p['tt_me'], p['tt_te']])
ii = np.concatenate([p['ii_me'], p['ii_te']])
pr = np.concatenate([p['p_lgb_me'], p['p_lgb_te']])
print('对照段 %s ~ %s | %d 日 | %d 行' %
      (dates[tt.min()].date(), dates[tt.max()].date(), len(np.unique(tt)), len(tt)), flush=True)

# 按日组织
day = {}
for t, s, v in zip(tt, ii, pr):
    day.setdefault(t, []).append((s, v))
day = {k: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for k, v in day.items()}
tlist = sorted(day.keys())


def run(mode, q=0.10, cost=0.0015, ind=None):
    """mode: 'all' 全市场剔底 | 'ind' 行业内剔底"""
    rows, prev = [], None
    prev_w = None
    for t in tlist[::STEP]:
        s, v = day[t]
        y = Y2[t, s]
        ok = np.isfinite(y)
        s, v, y = s[ok], v[ok], y[ok]
        if len(s) < 300: continue
        base = y.mean()                      # 当日全市场等权（=相对基准 0）
        n = max(int(len(s) * q), 10)
        if mode == 'all':
            bot = np.argsort(v)[:n]
        else:
            g = ind[t, s]
            bot = []
            for gid in np.unique(g):
                if gid < 0: continue
                idx = np.where(g == gid)[0]
                if len(idx) < 50: continue   # 行业太小不剔，避免单票冲击
                k = max(int(len(idx) * q), 3)
                bot.extend(idx[np.argsort(v[idx])[:k]])
            bot = np.array(bot, dtype=int)
        keep = np.ones(len(s), bool); keep[bot] = False
        w = keep.astype(float); w /= w.sum()
        r = float((w * y).sum() - base)      # 相对全市场等权的超额
        wf = np.zeros(N); wf[s] = w          # 按全局股票索引对齐
        to = 0.0 if prev_w is None else float(np.abs(wf - prev_w).sum() / 2)
        rows.append((t, r, to, int((~keep).sum())))
        prev_w = wf
    d = pd.DataFrame(rows, columns=['t', 'ex', 'to', 'n_bot'])
    d['net'] = d['ex'] - d['to'] * 2 * cost
    return d


print('\n%-26s %9s %9s %7s %8s %7s' % ('策略(单边15bp)', '年化超额', '波动', '夏普', '换手/年', '剔出数'))
res = []
for nm, mode, ind in [('全市场剔底10%', 'all', None),
                      ('行业内剔底10% · 申万', 'ind', IND_sw),
                      ('行业内剔底10% · 聚类', 'ind', IND_cl)]:
    d = run(mode, ind=ind)
    ann = d.net.mean() * PPY * 100
    vol = d.net.std() * np.sqrt(PPY) * 100
    print('%-26s %+8.2f%% %8.2f%% %7.2f %8.0f %7.0f' %
          (nm, ann, vol, ann / vol, d.to.mean() * PPY * 2, d.n_bot.mean()))
    res.append(dict(strategy=nm, ann=ann, vol=vol, sharpe=ann / vol,
                    turnover=d.to.mean() * PPY * 2, periods=len(d)))
    # 成本敏感性
    for c in [0.0, 0.0020]:
        dd = run(mode, cost=c, ind=ind)
        print('   %s 单边%.0fbp: 年化 %+.2f%%' %
              ('', c * 1e4, dd.net.mean() * PPY * 100), end='')
    print()

pd.DataFrame(res).to_csv(f'{ROOT}/reports/strategy_compare.csv', index=False)
print('\n已存 reports/strategy_compare.csv')

# 分年
print('\n=== 分年（单边 15bp，年化超额）===')
out = {}
for nm, mode, ind in [('全市场', 'all', None), ('行业内-申万', 'ind', IND_sw)]:
    d = run(mode, ind=ind)
    d['yr'] = pd.DatetimeIndex(dates[d.t]).year
    out[nm] = d.groupby('yr').net.apply(lambda x: x.mean() * PPY * 100)
print(pd.DataFrame(out).round(2).to_string())
