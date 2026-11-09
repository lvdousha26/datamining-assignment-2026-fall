# -*- coding: utf-8 -*-
"""策略轮动：在 A1 / A2 / 债券 之间按动量或风险动态配权

基线：固定 70/30（夏普 1.35）
变体：
  ① 动量轮动（过去 N 月收益排序，超配赢家）
  ② 反转轮动（超配输家）
  ③ 三资产 top1 / top2（A1/A2/债）
  ④ 风险平价（按波动倒数）
  ⑤ 等权 50/50
全部 walk-forward：月末用过去 N 月数据定下月权重。
"""
import io, contextlib, os, sys
import numpy as np, pandas as pd

ROOT, BASE = 'D:/quant', r'C:/Users/25097/WorkBuddy/2026-10-30-23-31-35'
src = open(f'{ROOT}/scripts/combine_A1_A2.py', encoding='utf-8').read()
exec(src.split('# ---------- A1 ----------')[0])

bp = pd.read_pickle(f'{BASE}/idx_000012.pkl')
bm = (1 + bp.pct_change()).resample('ME').prod() - 1
bm.index = bm.index.to_period('M')
os.chdir(BASE)
with contextlib.redirect_stdout(io.StringIO()):
    sys.path.insert(0, BASE)
    import spec3
os.chdir(ROOT)
g = spec3.legs['gold']; g.index = pd.PeriodIndex(g.index, freq='M')

j = pd.concat([a2_abs.rename('A2'), bm.rename('B'), g.rename('G')], axis=1).dropna()
rv = j.A2.rolling(6, min_periods=4).std() * 12 ** .5
w2 = (0.08 / rv).clip(0.2, 1.0).shift(1).fillna(0.5)
a2v = w2 * j.A2 + (1 - w2) * (0.7 * j.B + 0.3 * j.G)
a1 = spec3.s3B.dropna()

D = pd.concat([a1.rename('A1'), a2v.rename('A2')], axis=1).dropna()
B = bm.reindex(D.index).fillna(0)
print('对齐 %d 月 | %s ~ %s' % (len(D), D.index[0], D.index[-1]), flush=True)


def st(x, nm):
    a = (1 + x).prod() ** (12 / len(x)) - 1
    v = x.std() * 12 ** .5
    nav = (1 + x).cumprod(); dd = (nav / nav.cummax() - 1).min()
    print('  %-30s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%%' %
          (nm, a * 100, v * 100, (a - .02) / v, dd * 100))
    return (a - .02) / v


out = {}

print('\n=== 基线 ===')
out['固定70/30'] = st(0.7 * D.A1 + 0.3 * D.A2, '固定 70/30')
out['等权50/50'] = st(0.5 * D.A1 + 0.5 * D.A2, '等权 50/50')

print('\n=== ① 动量轮动（两策略）===')
for N in [3, 6, 9, 12]:
    mom = (1 + D).rolling(N).apply(np.prod, raw=True) - 1
    # 相对强度 → 权重（sigmoid 平滑），walk-forward 用 shift(1)
    diff = (mom.A2 - mom.A1).shift(1)
    wA2 = (0.5 + 0.5 * np.tanh(diff * 6)).clip(0, 1)     # 赢家权重 0-100%
    c = (1 - wA2) * D.A1 + wA2 * D.A2
    out['动量%d' % N] = st(c, '动量 N=%d（超配赢家）' % N)

print('\n=== ② 反转轮动 ===')
for N in [3, 6, 12]:
    mom = (1 + D).rolling(N).apply(np.prod, raw=True) - 1
    diff = (mom.A2 - mom.A1).shift(1)
    wA2 = (0.5 - 0.5 * np.tanh(diff * 6)).clip(0, 1)
    c = (1 - wA2) * D.A1 + wA2 * D.A2
    out['反转%d' % N] = st(c, '反转 N=%d（超配输家）' % N)

print('\n=== ③ 三资产 top1 / top2（A1/A2/债）===')
P = pd.concat([D.A1.rename('A1'), D.A2.rename('A2'), B.rename('B')], axis=1)
for N in [6, 12]:
    mom = (1 + P).rolling(N).apply(np.prod, raw=True) - 1
    for topk in [1, 2]:
        w = pd.DataFrame(0.0, index=P.index, columns=P.columns)
        for i in range(len(P)):
            if i == 0 or mom.iloc[i - 1].isna().any(): continue
            r = mom.iloc[i - 1].nlargest(topk)
            for cc in r.index:
                w.iloc[i, w.columns.get_loc(cc)] = 1.0 / topk
        c = (w * P).sum(1)
        out['top%d_N%d' % (topk, N)] = st(c, '动量 top%d N=%d' % (topk, N))

print('\n=== ④ 风险平价（两策略，按波动倒数）===')
vol = D.rolling(6, min_periods=4).std().shift(1)
iv = (1 / vol).div((1 / vol).sum(1), axis=0)
c = iv.A1 * D.A1 + iv.A2 * D.A2
out['风险平价'] = st(c, '风险平价（波动倒数）')
print('     平均 A2 权重 %.0f%%' % (iv.A2.mean() * 100))

print('\n=== 汇总 ===')
s = pd.Series(out).sort_values(ascending=False)
print(s.round(3).to_string())
print('\n基线（固定70/30）夏普 %.2f | 最优变体 %.2f（%s）| 提升 %+.2f' %
      (out['固定70/30'], s.iloc[0], s.index[0], s.iloc[0] - out['固定70/30']))
s.to_csv(f'{ROOT}/reports/strategy_rotation.csv')
print('已存 reports/strategy_rotation.csv')
