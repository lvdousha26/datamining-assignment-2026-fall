# -*- coding: utf-8 -*-
"""缠论结构因子：分型 → 笔 → 中枢 → 背驰，全部转成可计算的截面因子

缠论原文（《缠中说禅》）是交易体系，不是因子库。能落地的只有几何结构：
  1. **K 线包含处理**：有包含关系的两根合并（方向由前一笔决定）
  2. **分型**：合并后三根 K 线，中间的高点最高 = 顶分型；低点最低 = 底分型
  3. **笔**：顶底分型交替且间隔 ≥4 根合并 K 线，连成一笔
  4. **中枢**：连续三笔的重叠区间，ZG = min(三笔高点)、ZD = max(三笔低点)
  5. **背驰**：同向两笔里，价格创新极值但 MACD 面积缩小

由此构造 10 个因子（全部只用 t 时点及之前的数据，分型在下一根才确认、笔在下一根才成立）：
  chan_bi_dir      当前笔方向（+1 上行 / -1 下行）
  chan_bi_days     当前笔已走几根 K 线（越老越可能结束）
  chan_bi_amp      当前笔幅度（%）
  chan_bi_speed    当前笔速度（幅度/天数）
  chan_zs_pos      收盘价在中枢里的相对位置（<0 在下方，>1 在上方）
  chan_zs_width    中枢宽度（ZG-ZD）/ 中轴，衡量盘整的松紧
  chan_zs_n60      过去 60 日新形成的中枢个数（越少越趋势）
  chan_zs_break    向上突破中枢的强度（close>ZG 时 = 超出幅度/中枢宽度，否则 0）
  chan_div         近 10 日内的背驰信号（+1 底背驰 / -1 顶背驰 / 0）
  chan_fx_age      距最近一个分型几根 K 线

评估：日频截面 rank IC vs Y2（5 日前向超额），随机 1,500 只，与 wq101 同一口径。

用法：python scripts/chan_factors.py
产出：reports/chan_factors.md + .csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from wq101_factors import load_wide  # noqa: E402

ROOT = r'D:/quant'
N_SYM = 1500
MIN_GAP = 4          # 笔：两个分型中心至少间隔 4 根合并 K 线
NAMES = ['chan_bi_dir', 'chan_bi_days', 'chan_bi_amp', 'chan_bi_speed',
         'chan_zs_pos', 'chan_zs_width', 'chan_zs_n60', 'chan_zs_break',
         'chan_div', 'chan_fx_age']


def macd(c, fast=12, slow=26, sig=9):
    e1 = pd.Series(c).ewm(span=fast, adjust=False).mean()
    e2 = pd.Series(c).ewm(span=slow, adjust=False).mean()
    dif = e1 - e2
    dea = dif.ewm(span=sig, adjust=False).mean()
    return dif.values, (dif - dea).values * 2


def merge_bars(h, l):
    """K 线包含处理：返回合并后的 (高点, 低点, 原始→合并索引)"""
    n = len(h)
    mh = np.empty(n, np.float64)
    ml = np.empty(n, np.float64)
    mp = np.zeros(n, np.int32)
    k = -1
    trend = 1
    for i in range(n):
        if k < 0:
            k = 0
            mh[0] = h[i]
            ml[0] = l[i]
            mp[i] = 0
            continue
        inc = (h[i] <= mh[k] and l[i] >= ml[k]) or (h[i] >= mh[k] and l[i] <= ml[k])
        if inc:
            if trend >= 0:
                mh[k] = max(mh[k], h[i])
                ml[k] = max(ml[k], l[i])
            else:
                mh[k] = min(mh[k], h[i])
                ml[k] = min(ml[k], l[i])
        else:
            k += 1
            mh[k] = h[i]
            ml[k] = l[i]
            trend = 1 if h[i] > mh[k - 1] else -1
        mp[i] = k
    return mh[:k + 1], ml[:k + 1], mp


def fractals(mh, ml):
    """顶/底分型：返回 [(合并索引, +1 顶 / -1 底)]"""
    out = []
    for i in range(1, len(mh) - 1):
        if mh[i] > mh[i - 1] and mh[i] > mh[i + 1] and ml[i] > ml[i - 1] and ml[i] > ml[i + 1]:
            out.append((i, 1))
        elif ml[i] < ml[i - 1] and ml[i] < ml[i + 1] and mh[i] < mh[i - 1] and mh[i] < mh[i + 1]:
            out.append((i, -1))
    return out


def strokes(fx, mh, ml):
    """由分型连成笔：同向取更极端，异向且间隔够才成笔
    返回 [(起(i), 止(j), 方向, 高, 低, 确认于 j+1)]
    """
    out = []
    if not fx:
        return out
    last_i, last_t = fx[0]
    for i, t in fx[1:]:
        if t == last_t:
            # 同类型：保留更极端的（顶取更高、底取更低），但必须仍与上一笔端点间隔够
            if (t == 1 and mh[i] >= mh[last_i]) or (t == -1 and ml[i] <= ml[last_i]):
                last_i = i
            continue
        if i - last_i < MIN_GAP:
            # 间隔太近，不成笔；若新分型更极端则挪动上一个端点
            continue
        hi = float(np.max(mh[last_i:i + 1]))
        lo = float(np.min(ml[last_i:i + 1]))
        out.append((last_i, i, t, hi, lo, i + 1))
        last_i, last_t = i, t
    return out


def pivots(sts):
    """中枢：连续三笔的重叠。ZG = min(三笔高), ZD = max(三笔低)，ZG>ZD 才成立。
    返回 [(起始笔号, 结束笔号, ZG, ZD, 确认于第几根合并 K 线)]
    """
    out = []
    n = len(sts)
    i = 0
    while i + 2 < n:
        seg = sts[i:i + 3]
        zg = min(s[3] for s in seg)
        zd = max(s[4] for s in seg)
        if zg > zd:
            j = i + 2
            # 后续笔仍与 [ZD, ZG] 有重叠则中枢延伸
            while j + 1 < n and not (sts[j + 1][4] > zg or sts[j + 1][3] < zd):
                j += 1
            out.append((i, j, zg, zd, sts[j][5]))
            i = j + 1
        else:
            i += 1
    return out


def one_stock(o, h, l, c):
    """单只股票：返回 {因子名: 长度 n 的数组}（原始日频，全部因果）"""
    n = len(c)
    F = {k: np.full(n, np.nan, np.float32) for k in NAMES}
    if n < 60:
        return F
    ok = np.isfinite(c) & np.isfinite(h) & np.isfinite(l)
    if ok.sum() < 60:
        return F
    # 停牌日 ffill（结构在停牌期间"冻结"），最后把停牌位置置 NaN
    o = pd.Series(o).ffill().bfill().values
    h = pd.Series(h).ffill().bfill().values
    l = pd.Series(l).ffill().bfill().values
    c = pd.Series(c).ffill().bfill().values
    mh, ml, mp = merge_bars(h, l)
    M = len(mh)
    if M < 30:
        return F
    fx = fractals(mh, ml)
    sts = strokes(fx, mh, ml)
    zs = pivots(sts)

    # 合并 K 线 → 原始 K 线的映射（取每组合并 K 线的最后一根）
    last_orig = np.zeros(M, np.int32)
    for i in range(n):
        last_orig[mp[i]] = i
    mclose = c[last_orig]

    # 每根合并 K 线：当前笔起点 / 方向 / 已走天数
    bi_start = np.zeros(M, np.int32)
    bi_dir = np.zeros(M, np.int8)
    bi_hi = np.zeros(M)
    bi_lo = np.zeros(M)
    ptr = 0
    cur_s, cur_d = 0, 1
    cur_hi, cur_lo = mh[0], ml[0]
    for m in range(M):
        while ptr < len(sts) and sts[ptr][5] <= m:      # 该笔已确认
            s = sts[ptr]
            cur_s, cur_d = s[1], -s[2]      # 终点是顶 → 下一笔向下；终点是底 → 向上
            cur_hi, cur_lo = mh[s[1]], ml[s[1]]
            ptr += 1
        bi_start[m] = cur_s
        bi_dir[m] = cur_d
        bi_hi[m] = max(cur_hi, mh[m])
        bi_lo[m] = min(cur_lo, ml[m])

    # 中枢状态（因果：确认后才可用）
    zp = 0
    zcur, zconf = None, -10 ** 6
    zs_start = np.zeros(M, np.int32)
    zs_zg = np.full(M, np.nan)
    zs_zd = np.full(M, np.nan)
    zs_n60 = np.zeros(M, np.int32)
    zs_list_bar = []
    ZS_LIFE = 40          # 中枢确认后 40 根合并 K 线内仍算"当前中枢"，过期则视为已走远
    for m in range(M):
        while zp < len(zs) and zs[zp][4] <= m:
            zcur, zconf = zs[zp], zs[zp][4]
            zs_list_bar.append(m)
            zp += 1
        if zcur is not None and m - zconf <= ZS_LIFE:
            zs_start[m] = zcur[0]
            zs_zg[m] = zcur[2]
            zs_zd[m] = zcur[3]
        zs_n60[m] = sum(1 for b in zs_list_bar if b > m - 60)

    # 背驰：同向两笔比 MACD 面积（价格创新极值但面积缩小）
    dif, hist = macd(c)
    div = np.zeros(M, np.float32)
    for k in range(2, len(sts)):
        s, prev = sts[k], sts[k - 2]
        if s[2] != prev[2]:
            continue
        a0, a1 = last_orig[prev[0]], last_orig[prev[1]]
        b0, b1 = last_orig[s[0]], last_orig[s[1]]
        if b1 <= b0 or a1 <= a0:
            continue
        area_new = np.nansum(np.abs(hist[b0:b1 + 1]))
        area_old = np.nansum(np.abs(hist[a0:a1 + 1]))
        if s[2] == -1 and s[4] < prev[4] and area_new < area_old * 0.9:
            div[s[1]] = 1.0        # 底背驰：新低但力度弱
        elif s[2] == 1 and s[3] > prev[3] and area_new < area_old * 0.9:
            div[s[1]] = -1.0       # 顶背驰

    # ---- 因果化：状态统一滞后一根合并 K 线 ----
    # 分型/笔/中枢都要等「下一根合并 K 线走完」才确认，而一根合并 K 线含多根日线。
    # 严格做法是按组内位置逐个判断；这里统一滞后一根，宁可保守（约 5 个交易日）。
    def lag1(a, fill):
        out = np.empty_like(a)
        out[0] = fill
        out[1:] = a[:-1]
        return out

    bi_start = lag1(bi_start, 0)
    bi_dir = lag1(bi_dir, 0)
    zs_zg = lag1(zs_zg, np.nan)
    zs_zd = lag1(zs_zd, np.nan)
    zs_n60 = lag1(zs_n60, 0)
    div = lag1(div, 0.0)

    # ---- 落到原始日频 ----
    m_idx = mp
    bs = bi_start[m_idx]
    # 当前笔天数 / 幅度 / 速度（用原始 K 线计数，更符合"几根日线"的直觉）
    days = (np.arange(n) - last_orig[bs]).astype(np.float32)
    start_px = mclose[bs]
    amp = (c - start_px) / np.maximum(start_px, 1e-9)
    F['chan_bi_dir'] = bi_dir[m_idx].astype(np.float32)
    F['chan_bi_days'] = days
    F['chan_bi_amp'] = amp
    F['chan_bi_speed'] = amp / np.maximum(days, 1.0)

    zg = zs_zg[m_idx]
    zd = zs_zd[m_idx]
    w = zg - zd
    pos = np.where(np.isfinite(w) & (w > 0), (c - zd) / np.maximum(w, 1e-9), np.nan)
    F['chan_zs_pos'] = np.clip(pos - 0.5, -3, 4).astype(np.float32)
    F['chan_zs_width'] = np.where(np.isfinite(w) & (w > 0),
                                  w / np.maximum((zg + zd) / 2, 1e-9), np.nan).astype(np.float32)
    F['chan_zs_n60'] = zs_n60[m_idx].astype(np.float32)
    hasz = np.isfinite(w) & (w > 0)
    brk = np.where(hasz, np.where(c > zg, (c - zg) / np.maximum(w, 1e-9), 0.0), np.nan)
    F['chan_zs_break'] = np.clip(brk, 0, 5).astype(np.float32)

    # 背驰：近 10 个交易日内的信号
    dz = div[m_idx]
    out = np.zeros(n, np.float32)
    last_v, last_i = 0.0, -10 ** 6
    for i in range(n):
        if dz[i] != 0:
            last_v, last_i = float(dz[i]), i
        if i - last_i <= 10:
            out[i] = last_v
    F['chan_div'] = out

    # 最近分型距今几根
    fb = np.full(n, -1, np.int32)
    last_fx = -1
    fxi = 0
    # 分型要等下一根合并 K 线走完才确认：记录（确认日, 分型所在日）
    fx_con = sorted((last_orig[min(i + 1, M - 1)], last_orig[i]) for i, _ in fx)

    def _fx_age():
        nonlocal last_fx, fxi
        for i in range(n):
            while fxi < len(fx_con) and fx_con[fxi][0] < i:
                last_fx = fx_con[fxi][1]
                fxi += 1
            fb[i] = i - last_fx if last_fx >= 0 else 99
    _fx_age()
    F['chan_fx_age'] = fb.astype(np.float32)
    for k in NAMES:
        F[k][~ok] = np.nan
    return F


def main():
    z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
    Y2 = z['Y2']
    dates = pd.to_datetime(pd.Series(z['dates']).astype(str))
    syms_all = [str(s).zfill(6) for s in z['symbols']]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(syms_all, min(N_SYM, len(syms_all)), replace=False))
    ci = [syms_all.index(s) for s in pick]

    P = load_wide(pick)
    o, h, l, c = P['open'].values, P['high'].values, P['low'].values, P['close'].values
    pdate = pd.to_datetime(P['close'].index)
    print('  样本 %d 只 × %d 日' % (c.shape[1], c.shape[0]), flush=True)

    t0 = time.time()
    mats = {k: np.full_like(Y2, np.nan) for k in NAMES}
    rows = []
    for j, s in enumerate(pick):
        Fj = one_stock(o[:, j], h[:, j], l[:, j], c[:, j])
        for k in NAMES:
            v = pd.Series(Fj[k], index=pdate).reindex(dates).values
            mats[k][:, ci[j]] = v
        if (j + 1) % 300 == 0:
            print('    %d/%d 只（%.0fs）' % (j + 1, len(pick), time.time() - t0), flush=True)
    print('  结构计算完成 %.0fs' % (time.time() - t0), flush=True)

    # ---- IC 评估 ----
    tr = np.where((dates.dt.year >= 2016) & (dates.dt.year <= 2024))[0]
    res = []
    for k in NAMES:
        f = mats[k]
        ics = []
        for t in tr:
            m = np.isfinite(f[t]) & np.isfinite(Y2[t])
            if m.sum() > 300:
                ics.append(spearmanr(f[t][m], Y2[t][m]).statistic)
        s = pd.Series(ics).dropna()
        if len(s) < 100:
            continue
        res.append(dict(factor=k, ic=s.mean(), icir=s.mean() / s.std() if s.std() else np.nan,
                        win=(s > 0).mean(), days=len(s),
                        cover=float(np.isfinite(f[tr]).mean())))
    r = pd.DataFrame(res).sort_values('ic', key=abs, ascending=False)
    print('\n' + r.round(4).to_string(index=False), flush=True)
    r.to_csv(f'{ROOT}/reports/chan_factors.csv', index=False)
    print('\n→', f'{ROOT}/reports/chan_factors.csv')


if __name__ == '__main__':
    with RunLogger(__file__, tag='chan_factors'):
        main()
