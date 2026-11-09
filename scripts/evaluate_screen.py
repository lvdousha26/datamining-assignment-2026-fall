# -*- coding: utf-8 -*-
"""禁买清单评估（lockbox 已消耗，只能做段内 + 折内评估）

评估三件事：
  1. dev 段（2025H1，111 日）剔除垫底 10% 的超额、波动、夏普、换手
  2. meta 段（2024，236 日）同样口径，看是否稳定
  3. 分模型对比：LGBM / TF / 因子合成 / Ridge 元模型

口径：
  - 收益用 Y2（t 日信号 → t+1 收盘买入 → 持有 5 日，且已截面去均值），
    所以「保留部分的 Y2 均值」就是相对全市场等权的超额，不需要再减基准
  - 5 日调仓，单边 15bp，换手按 D1 名单更替率计
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen_stats import d1_churn  # noqa: E402  名单更替率（只认股票索引 i）

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
z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
Y2, M, dates = z['Y2'], z['M'], pd.to_datetime(z['dates'])
p = np.load(f'{ROOT}/data/big/preds.npz')
COST = 0.0015
PPY = 252 / 5


def evaluate(pred, tt, ii, tag, q=0.10, step=5):
    # i 是股票在面板里的列索引，跨日稳定；更替率统一交给 screen_stats.d1_churn
    # （曾经用 DataFrame 行索引算，跨日不重叠 → 每期恒 100%，2026-11-09 修）
    d = pd.DataFrame({'t': tt, 'i': ii, 'p': pred, 'y': Y2[tt, ii]}).dropna()
    churn = {t: (n, c) for t, n, c in d1_churn(d, q=q, step=step)}
    rows = []
    for t, (n, to) in churn.items():
        gg = d[d['t'] == t]
        bot = gg.nsmallest(n, 'p')
        keep = gg.drop(bot.index)
        ex = keep['y'].mean() - gg['y'].mean()
        rows.append((t, ex, to, n))
    r = pd.DataFrame(rows, columns=['t', 'ex', 'to', 'n']).dropna()
    r['net'] = r['ex'] - r['to'] * q * 2 * COST      # 只有被替换的 q 份额产生交易
    ann = (1 + r['net']).prod() ** (PPY / len(r)) - 1
    vol = r['net'].std() * np.sqrt(PPY)
    nav = (1 + r['net']).cumprod()
    dd = (nav / nav.cummax() - 1).min()
    print('  %-16s 年化超额 %+6.2f%% | 波动 %5.2f%% | 超额夏普 %5.2f | 回撤 %6.2f%% | 期数 %3d | D1换手 %3.0f%%'
          % (tag, ann * 100, vol * 100, ann / vol if vol > 0 else np.nan,
             dd * 100, len(r), r['to'].mean() * 100))
    return dict(tag=tag, ann=float(ann), vol=float(vol), n=int(len(r)),
                turnover=float(r['to'].mean()))


print('=== 禁买清单评估（剔除打分垫底 10%，5 日调仓，单边 15bp）===')
print('  注：Y2 已截面去均值，故「保留部分均值」= 相对全市场等权的超额\n')
out = []

for seg, key in [('meta 段 2024', 'me'), ('dev 段 2025H1', 'te')]:
    tt, ii = p[f'tt_{key}'], p[f'ii_{key}']
    nd = len(np.unique(tt))
    print('%s（%d 日，%s ~ %s）' % (seg, nd, dates[tt.min()].date(), dates[tt.max()].date()))
    for name, col in [('LGBM', f'p_lgb_{key}'), ('TF', f'p_tf_{key}'),
                      ('因子合成', f'fac_{key}'), ('Ridge 元模型', f'p_st_{key}')]:
        if col in p.files:
            out.append(evaluate(p[col], tt, ii, name))
    print()

# 分年（用 meta+dev 拼起来）
tt = np.concatenate([p['tt_me'], p['tt_te']])
ii = np.concatenate([p['ii_me'], p['ii_te']])
pr = np.concatenate([p['p_st_me'], p['p_st_te']])
d = pd.DataFrame({'t': tt, 'i': ii, 'p': pr, 'y': Y2[tt, ii]}).dropna()
d['yr'] = dates[d['t']].year
print('分年（Ridge 元模型，年化超额，扣成本）：')
for y, gg in d.groupby('yr'):
    churn = {t: (n, c) for t, n, c in d1_churn(gg, q=0.10, step=5)}
    rows = []
    for t, (n, to) in churn.items():
        g = gg[gg['t'] == t]
        b = g.nsmallest(n, 'p')
        keep = g.drop(b.index)
        ex = keep['y'].mean() - g['y'].mean()
        rows.append(ex - to * 0.10 * 2 * COST)
    if rows:
        rows = np.array(rows)
        print('  %d: %+6.2f%% /年（%d 期）' % (y, ((1 + rows).prod() ** (PPY / len(rows)) - 1) * 100, len(rows)))

json.dump(out, open(f'{ROOT}/reports/screen_eval.json', 'w', newline='\n'),
          ensure_ascii=False, indent=1)
print('\n已存 reports/screen_eval.json')
