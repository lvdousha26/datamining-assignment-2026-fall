# -*- coding: utf-8 -*-
"""复现"行业动量追涨 + 盈亏加减仓"热点短线策略（截图描述）— 修正口径版

三个口径要点：
1. 收益用 Y2（已截面去均值）→ 组合的 Y2 均值本身就是"相对全市场等权的超额"；
   年化用 算术均值×50.4，避免几何年化里的波动拖累污染横向比较。
2. 成本按持仓权重向量的单边更替率算；仓位缩放不算换手（那只是现金/融资比例变化）。
3. 粒度对齐：申万二级 124 个里选前 10 ≈ 前 8%。我们 31 组选前 3（=9.7%）才是等价口径，
   所以 topk=3 是主口径，其余只作敏感性。
"""
import numpy as np, pandas as pd

ROOT = 'D:/quant'
COST = 0.0015
STEP = 5
PPY = 252 / STEP

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, Y2, M, dates, symbols = z['X'], z['Y2'], z['M'], z['dates'], z['symbols']
T, N = X.shape[:2]
fi = z['feats'].tolist().index('ret20')
ind = pd.read_parquet(f'{ROOT}/data/big/ind_roll.parquet')
y2c = {(int(r.year), str(r.symbol)): int(r.cluster) for r in ind.itertuples()}
sym = [str(s) for s in symbols]
yrs = pd.DatetimeIndex(dates).year.values
CID = np.full((T, N), -1, np.int32)
for j, s in enumerate(sym):
    for t in range(T):
        c = y2c.get((int(yrs[t]), s))
        if c is not None: CID[t, j] = c
NC = int(ind['cluster'].max()) + 1


def snapshots():
    """预计算每个调仓日的行业动量排序与全市场基准"""
    snaps = []
    for t0 in range(0, T - 7, STEP):
        t = t0 + 1
        m = M[t] & np.isfinite(Y2[t]) & (CID[t] >= 0)
        if m.sum() < 300: continue
        r = X[t, :, fi].astype(float)
        ok = m & np.isfinite(r)
        rr = r[ok]
        zc = (rr - rr.mean()) / (rr.std() + 1e-12)
        cid = CID[t][ok]
        cnt = np.bincount(cid, minlength=NC).astype(float)
        sm = np.bincount(cid, weights=zc, minlength=NC)
        mom = np.where(cnt > 20, sm / np.maximum(cnt, 1), np.nan)
        # 下期行业等权收益（用于行业层 IC）
        yv = Y2[t][ok]
        sy = np.bincount(cid, weights=np.nan_to_num(yv), minlength=NC)
        ir = np.where(cnt > 20, sy / np.maximum(cnt, 1), np.nan)
        base = Y2[t][M[t] & np.isfinite(Y2[t])].mean()
        snaps.append(dict(t=t, m=m, mom=mom, ir=ir, base=base))
    return snaps


snaps = snapshots()
print('调仓期数 %d | %s ~ %s' % (len(snaps), dates[snaps[0]['t']],
                                  dates[snaps[-1]['t']]))

# ---------- ① 行业动量 IC（31 组截面秩相关，逐年）----------
from scipy import stats as sps
ic_rows = []
for s in snaps:
    ok = np.isfinite(s['mom']) & np.isfinite(s['ir'])
    if ok.sum() < 15: continue
    ic_rows.append((s['t'], sps.spearmanr(s['mom'][ok], s['ir'][ok]).correlation))
icd = pd.DataFrame(ic_rows, columns=['t', 'ic'])
icd['year'] = pd.DatetimeIndex(dates[icd['t']]).year
byyear_ic = icd.groupby('year')['ic'].agg(['mean', 'count'])
print('\n=== ① 行业动量（过去20日行业涨幅 → 下5日行业收益）逐年秩相关 ===')
print(byyear_ic.round(3).to_string())
print('全样本 IC 均值 %+.4f | ICIR %.2f | 正 IC 占比 %.0f%%' %
      (icd.ic.mean(), icd.ic.mean()/icd.ic.std(), (icd.ic > 0).mean()*100))


def run(topk, reverse=False, sizing=False, cost=COST):
    rows = []; prev = None; pos = 1.0
    for s in snaps:
        t = s['t']
        mom = s['mom']
        order = np.argsort(np.where(np.isfinite(mom), mom, -np.inf))
        sel = order[:topk] if reverse else order[-topk:]
        hold = np.isin(CID[t], sel) & M[t] & np.isfinite(Y2[t])
        unit = hold.astype(float); unit /= unit.sum()
        w = unit * pos
        g = float(np.nansum(w * np.nan_to_num(Y2[t])))
        # 换手：只算单位组合的权重更替（仓位缩放不计）
        to = 0.0 if prev is None else float(np.abs(unit - prev).sum() / 2)
        rows.append((t, g - s['base'] * pos, to, pos, int(hold.sum())))
        if sizing:
            pos = min(max(pos * 2 if g > 0.03 else (pos * .5 if g < -0.03 else pos), 0.0), 1.0)
        prev = unit
    d = pd.DataFrame(rows, columns=['t', 'ex', 'to', 'pos', 'n'])
    d['net'] = d['ex'] - d['to'] * 2 * cost * d['pos']
    d['year'] = pd.DatetimeIndex(dates[d['t']]).year
    return d


def stat(d):
    return dict(n=len(d), ex=d.ex.mean()*PPY*100, net=d.net.mean()*PPY*100,
                sharpe=d.ex.mean()/d.ex.std()*np.sqrt(PPY),
                to=d.to.mean()*PPY*2, pos=d.pos.mean())


print('\n=== ② 各口径：年化超额（相对全市场等权，算术） ===')
rows = []
for k in [3, 5, 8, 10]:
    rows.append(dict(策略='追涨 top%d' % k, **stat(run(k))))
    rows.append(dict(策略='追跌 top%d' % k, **stat(run(k, reverse=True))))
    rows.append(dict(策略='追涨 top%d + 仓位规则' % k, **stat(run(k, sizing=True))))
res = pd.DataFrame(rows)
print(res.round(2).to_string(index=False))

print('\n=== ③ 分年：追涨 top3（等价申万前8%）vs 追跌 top3，毛超额 ===')
a = run(3); b = run(3, reverse=True)
py = pd.DataFrame({'追涨': a.groupby('year').ex.apply(lambda x: x.mean()*PPY*100),
                   '追跌': b.groupby('year').ex.apply(lambda x: x.mean()*PPY*100),
                   '期数': a.groupby('year').ex.count()})
print(py.round(1).to_string())

print('\n=== ④ 仓位规则到底值不值：同平均仓位对照 ===')
for k in [3, 10]:
    s1 = run(k, sizing=True); s0 = run(k)
    p = s1.pos.mean()
    fixed = s0.ex * p              # 固定仓位 = 仓位规则的平均仓位
    print('top%-2d 仓位规则 年化 %+.2f%% / 夏普 %.2f / 平均仓位 %.0f%%  ||  '
          '固定同仓位 年化 %+.2f%% / 夏普 %.2f  ||  差 %+.2f%%'
          % (k, s1.ex.mean()*PPY*100, s1.ex.mean()/s1.ex.std()*np.sqrt(PPY), p*100,
             fixed.mean()*PPY*100, fixed.mean()/fixed.std()*np.sqrt(PPY),
             (s1.ex.mean()-fixed.mean())*PPY*100))

print('\n=== ⑤ 成本敏感性：毛 / 0bp / 10bp / 15bp（年化超额%） ===')
for k in [3, 8, 10]:
    for rev in (False, True):
        d0 = run(k, reverse=rev, cost=0.0); d1 = run(k, reverse=rev, cost=0.0010)
        d2 = run(k, reverse=rev)
        print('top%-2d %-4s 毛 %+6.2f | 0bp %+6.2f | 10bp %+6.2f | 15bp %+6.2f | 换手 %.0f倍/年'
              % (k, '追跌' if rev else '追涨', d0.ex.mean()*PPY*100, d0.net.mean()*PPY*100,
                 d1.net.mean()*PPY*100, d2.net.mean()*PPY*100, d0.to.mean()*PPY*2))

res.to_csv(f'{ROOT}/reports/hot_sector_scan.csv', index=False)
py.round(3).to_csv(f'{ROOT}/reports/hot_sector_byyear.csv')
icd.to_csv(f'{ROOT}/reports/hot_sector_ind_ic.csv', index=False)
print('\n已存 hot_sector_scan.csv / hot_sector_byyear.csv / hot_sector_ind_ic.csv')
