# -*- coding: utf-8 -*-
"""多因子预测模型体检（只碰 dev/2025H1，绝不触碰封存段 2025-07 之后）

回答三个问题：
  1) 模型打分在**有标签的验证段**预测力如何（IC / ICIR / t / 十分位单调性）
  2) 当期（面板最后一日）打分长什么样（分布、D1 名单、与上期清单的一致性）
  3) LGBM / TF / 因子合成 三个输入各自的贡献

不重训、不调参、不评估封存段。用 `python scripts/probe_multifactor.py`
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from runlog import RunLogger

ROOT = r'D:/quant'


def ic_stats(tt, p, y):
    """按日截面 rank IC → 均值 / 标准差 / ICIR / t / 胜率"""
    df = pd.DataFrame({'t': tt, 'p': p, 'y': y}).dropna()
    rows = []
    for t, g in df.groupby('t'):
        if len(g) < 100 or g['y'].std() == 0:
            continue
        rows.append(spearmanr(g['p'], g['y']).statistic)
    s = pd.Series(rows).dropna()
    n = len(s)
    return {
        'n_days': n,
        'ic': s.mean(),
        'ic_std': s.std(),
        'icir': s.mean() / s.std() if s.std() else np.nan,
        't': s.mean() / (s.std() / np.sqrt(n)) if s.std() else np.nan,
        'win': (s > 0).mean(),
    }


def decile_table(tt, p, y, n=10):
    df = pd.DataFrame({'t': tt, 'p': p, 'y': y}).dropna()
    df['d'] = df.groupby('t')['p'].transform(lambda x: pd.qcut(x, n, labels=False, duplicates='drop'))
    g = df.groupby('d')['y'].agg(['mean', 'count'])
    g.index = g.index + 1
    return g


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    dates = pd.to_datetime(z['dates'])
    Y2 = z['Y2']
    lock_start = pd.Timestamp(str(np.atleast_1d(z['lock_start'])[0]))
    pm = np.load(f'{ROOT}/data/big/preds.npz')
    meta = json.load(open(f'{ROOT}/reports/big_models_meta.json'))

    print('=' * 72)
    print('多因子预测模型体检 | 封存段起点 %s（本次不评估其后的任何日期）' % lock_start.date())
    print('=' * 72)

    # ---------- 1. 各段预测力 ----------
    print('\n[1] 分段预测力（Y2 = 5 日前向收益，截面去均值）')
    print('%-26s %6s %8s %7s %7s %7s %6s' % ('输入/区间', '天数', 'IC', 'ICStd', 'ICIR', 't', '胜率'))
    segs = [('me', '2024 验证段'), ('va', '2025H1 dev'), ('te', '2025H1 test')]
    for key, name in [('p_st', 'Stacking'), ('p_lgb', 'LGBM'), ('p_tf', 'TF(3seed)'), ('fac', '因子合成')]:
        for seg, segname in segs:
            tt, ii, p = pm[f'tt_{seg}'], pm[f'ii_{seg}'], pm[key + '_' + seg]
            y = Y2[tt, ii]
            s = ic_stats(tt, p, y)
            print('%-26s %6d %8.4f %7.4f %7.3f %7.2f %6.1f%%' % (
                f'{key} @ {segname}', s['n_days'], s['ic'], s['ic_std'], s['icir'], s['t'], s['win'] * 100))

    # ---------- 2. 十分位单调性（dev 段） ----------
    tt, ii, p = pm['tt_va'], pm['ii_va'], pm['p_st_va']
    dec = decile_table(tt, p, Y2[tt, ii])
    print('\n[2] 2025H1 dev 段十分位（D1=打分最低，D10=最高；数值为 5 日收益，已去截面均值）')
    print('%-4s %9s %9s' % ('组', '5日均值', '股票数'))
    for d, r in dec.iterrows():
        print('D%-3d %8.3f%% %9d' % (d, r['mean'] * 100, r['count']))
    print('D10-D1 价差 = %.2f%%（5 日）' % ((dec.loc[10, 'mean'] - dec.loc[1, 'mean']) * 100))

    # ---------- 3. 当期截面 ----------
    print('\n[3] 当期打分（面板最后一日）')
    f = f'{ROOT}/live/screen_{dates[-1].date()}.csv'
    d = pd.read_csv(f, dtype={'symbol': str})
    print('文件 %s | %d 只 | 打分日 %s' % (f.split('/')[-1], len(d), d['date'].iloc[0]))
    print('score: min %.3f | p25 %.3f | 中位 %.3f | p75 %.3f | max %.3f' % (
        d.score.min(), d.score.quantile(.25), d.score.median(), d.score.quantile(.75), d.score.max()))
    print('D1（禁买）= %d 只 | D10 = %d 只' % (d.is_d1.sum(), (d.decile == 10).sum()))

    # 与上期清单一致性
    prev = f'{ROOT}/live/screen_2026-10-22.csv'
    try:
        q = pd.read_csv(prev, dtype={'symbol': str})
        j = d[['symbol', 'score']].merge(q[['symbol', 'score']], on='symbol', suffixes=('_new', '_old'))
        rho = spearmanr(j.score_new, j.score_old).statistic
        d1_new, d1_old = set(d.loc[d.is_d1 == 1, 'symbol']), set(q.loc[q.is_d1 == 1, 'symbol'])
        print('\n与 2026-10-22 清单对比: 共同股票 %d | 打分秩相关 %.4f | D1 重叠 %d/%d = %.1f%%' % (
            len(j), rho, len(d1_new & d1_old), len(d1_new), len(d1_new & d1_old) / len(d1_new) * 100))
    except Exception as e:
        print('（上期清单对比跳过：%s）' % e)

    print('\nD1 最垫底 10 只（禁买，仅列代码/名称/行业）：')
    print(d.nsmallest(10, 'score')[['symbol', 'name', 'industry', 'pct']].to_string(index=False))

    # ---------- 4. 同一模型、不同日期的稳定性 ----------
    # 注意：2026-10-22 在封存段内，这里只用特征打分、不碰任何标签，属于"当期信号"而非"验证"
    print('\n[4] 同一模型在相隔 6 个交易日上的稳定性（只打分，不看标签）')
    try:
        import lightgbm as lgb
        import torch
        from tf_net import TFNet, L
        X, M = z['X'], z['M']
        k = prior = None
        meta_w = np.array([meta['weights'].get(str(f), 0.0) for f in meta['feature_order']])
        coef = np.array(meta['coef'])
        MU, SD = meta['coef_mu'][:3], meta['coef_sd'][:3]
        booster = lgb.Booster(model_file=f'{ROOT}/checkpoints/lgbm_big.txt')
        lmeta = json.load(open(f'{ROOT}/checkpoints/lgbm_meta.json'))
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        nets = []
        for s in (1, 2, 3):
            m = TFNet(X.shape[2]).to(dev)
            m.load_state_dict(torch.load(f'{ROOT}/checkpoints/tf_seed{s}.pt', map_location=dev,
                                         weights_only=True))
            m.eval()
            nets.append(m)

        def score_at(t):
            ii = np.where(M[t])[0]
            pl = booster.predict(X[t, ii, :], num_iteration=lmeta['best_iter'])
            idx = np.full(len(ii), t)[:, None] - np.arange(L - 1, -1, -1)[None, :]
            xb = torch.from_numpy(X[idx, ii[:, None], :].astype(np.float32)).to(dev)
            pt = np.zeros(len(ii))
            with torch.no_grad():
                for m in nets:
                    pt += m(xb).cpu().numpy() / 3
            blk = X[t]
            r = np.full((X.shape[1], X.shape[2]), np.nan, np.float32)
            sub = blk[M[t]]
            r[M[t]] = ((sub.argsort(0).argsort(0) + .5) / len(sub)).astype(np.float32)
            fa = (np.nan_to_num(r) @ meta_w)[ii]
            A = np.column_stack([(pl - MU[0]) / SD[0], (pt - MU[1]) / SD[1],
                                 (fa - MU[2]) / SD[2], np.ones(len(ii))])
            return pd.DataFrame({'symbol': np.array([str(s) for s in z['symbols']])[ii],
                                 'score': A @ coef})

        d0 = score_at(len(dates) - 1)
        d6 = score_at(len(dates) - 7)
        j = d0.merge(d6, on='symbol', suffixes=('_new', '_old'))
        rho = spearmanr(j.score_new, j.score_old).statistic
        n = max(1, int(round(len(d0) * 0.1)))
        s1 = set(d0.nsmallest(n, 'score').symbol)
        s2 = set(d6.nsmallest(n, 'score').symbol)
        print('  %s vs %s：共同股票 %d | 打分秩相关 %.4f' % (
            dates[-1].date(), dates[-7].date(), len(j), rho))
        print('  D1（%d 只）重叠 %d = %.1f%%  → 6 日换手约 %.1f%%（每 5 日调仓）' % (
            n, len(s1 & s2), len(s1 & s2) / n * 100, (1 - len(s1 & s2) / n) * 100))
    except Exception as e:
        print('  （稳定性检查跳过：%s: %s）' % (type(e).__name__, e))

    # ---------- 5. 三个输入的权重 ----------
    print('\n[5] Stacking 元模型权重（冻结值）')
    print('  coef = %s' % np.round(meta['coef'], 4))
    print('  → LGBM / TF / 因子合成 三个输入的相对贡献，正负号即方向')


if __name__ == '__main__':
    with RunLogger(__file__, tag='probe'):
        main()
