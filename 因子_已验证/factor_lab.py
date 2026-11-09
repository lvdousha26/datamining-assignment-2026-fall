"""研报 -> 假设 -> 算子 -> 因子：实验室
算子类别: 基本运算 / 各阶矩 / 相关性 / 图关系 / 频谱 / 掩码
数据: a_ohlc.pkl (2096只, 2018-2026, OHLCV)
"""
import pickle, os
import numpy as np, pandas as pd

CACHE = 'hlmv.pkl'
if os.path.exists(CACHE):
    O, H, L, C, V = pickle.load(open(CACHE, 'rb'))
    print('缓存加载 OK', C.shape)
else:
    D = pickle.load(open('a_ohlc.pkl', 'rb'))
    def mk(i):
        return pd.DataFrame({c: pd.Series({pd.Timestamp(x[0]): x[i] for x in v})
                             for c, v in D.items()}).sort_index()
    O, H, L, C, V = mk(1), mk(2), mk(3), mk(4), mk(5)
    pickle.dump((O, H, L, C, V), open(CACHE, 'wb'))
    print('构建完成', C.shape)

C = C.loc[:, C.notna().sum() > 1200]
O, H, L, V = O[C.columns], H[C.columns], L[C.columns], V[C.columns]
print('有效股票 %d, 日期 %s ~ %s, %d 天' % (C.shape[1], C.index[0].date(), C.index[-1].date(), len(C)))

ret = C.pct_change()
gap = O / C.shift(1) - 1                       # 隔夜
intra = C / O - 1                              # 日内
mkt = ret.mean(axis=1)                         # 等权市场
vr = V / V.rolling(20).mean()                  # 量比

def rcorr(a, b, n):
    """滚动相关（向量化，比 DataFrame.rolling.corr 快得多）"""
    ca = a.sub(a.rolling(n).mean())
    cb = b.sub(b.rolling(n).mean())
    return (ca.mul(cb)).rolling(n).mean() / (a.rolling(n).std() * b.rolling(n).std() + 1e-12)

# ---------- 构造 18 个因子 ----------
F = {}

# 1. 矩类（研报语言：彩票偏好 / 尾部风险 / 极端收益）
F['SKEW20'] = ret.rolling(20).skew()                    # 偏度
F['KURT60'] = ret.rolling(60).kurt()                    # 峰度
F['MAXRET20'] = ret.rolling(20).max()                   # Bali MAX 效应
F['MINRET20'] = ret.rolling(20).min()                   # 极端下跌

# 2. 日内结构（研报语言：隔夜-日内收益分解 / 上影线卖压 / 收盘位置）
F['OVERNIGHT20'] = gap.rolling(20).sum()
F['INTRADAY20'] = intra.rolling(20).sum()
F['UPPERSH20'] = ((H - np.maximum(O, C)) / C).rolling(20).mean()
F['CLV20'] = ((C - L) / (H - L + 1e-9)).rolling(20).mean()

# 3. 量价（研报语言：量价背离 / Amihud非流动性 / 量能稳定性）
F['CORRPV20'] = rcorr(ret, V.pct_change(), 20)          # 收益与成交量变动相关
F['AMIHUD20'] = (ret.abs() / (V * C / 1e8)).rolling(20).mean()
F['VOLSTAB'] = V.rolling(20).mean() / V.rolling(250).mean()

# 4. 掩码类（研报语言：涨停基因 / 高位放量滞涨 / 跳空未回补）
cols20 = [c for c in C.columns if c.startswith(('30', '68'))]
fac = pd.DataFrame(1.1, index=C.index, columns=C.columns)
fac.loc[C.index >= '2020-08-24', cols20] = 1.2
limpx = (C.shift(1) * fac).round(2)
islim = (C - limpx).abs() < 0.011
F['LIMITUP60'] = islim.astype(float).rolling(60).sum()
F['VOLNOSPIKE'] = ((vr > 1.5) & (ret < 0.01)).astype(float).rolling(20).mean()
F['GAPABS20'] = gap.abs().rolling(20).mean()

# 5. 频谱 / 熵（研报语言：收益序列周期性 / 走势随机性）
p_up = (ret > 0).astype(float).rolling(20).mean()
F['ENTROPY20'] = -(p_up * np.log(p_up + 1e-9) + (1 - p_up) * np.log(1 - p_up + 1e-9))
ac1 = rcorr(ret, ret.shift(1), 60)
ac5 = rcorr(ret, ret.shift(5), 60)
F['PERIOD60'] = ac1.abs() + ac5.abs()                   # 周期性强度(FFT主频能量的代理)

# 6. 图关系（研报语言：市场联动 / 网络中心度）
F['CORRMKT60'] = rcorr(ret, mkt.to_frame().reindex(C.index).iloc[:, 0], 60)

print('因子 %d 个: %s' % (len(F), ', '.join(F.keys())))

# ---------- 月末截面 + 下月收益 ----------
me = C.groupby([C.index.year, C.index.month]).tail(1).index
me = me[me >= '2019-01-01']
nxt = C.reindex(me).pct_change().shift(-1)              # 月末到下月末
nxt.index = me

rows = []
for name, f in F.items():
    fm = f.reindex(me)
    ics, ls, monot, turn_prev = [], [], [], None
    for i in range(len(me) - 1):
        d = me[i]
        a, b = fm.loc[d], nxt.loc[d]
        m = a.notna() & b.notna() & np.isfinite(a)
        if m.sum() < 300: continue
        x, y = a[m], b[m]
        ics.append(x.corr(y, method='spearman'))
        q = pd.qcut(x.rank(method='first'), 10, labels=False)
        g = y.groupby(q).mean()
        ls.append(g.iloc[-1] - g.iloc[0])
        if name in ('__first__',): pass
        monot.append(np.corrcoef(range(10), g.reindex(range(10)).values)[0, 1])
        # 换手：最高组名单变动
        top = set(x[q == 9].index)
        if turn_prev is not None:
            turn_prev.append(len(top - turn_prev[-1]) / max(len(top), 1) if False else 0)
    ics = pd.Series(ics).dropna()
    ls = pd.Series(ls).dropna()
    if len(ics) < 20: continue
    ic_mean, ic_std = ics.mean(), ics.std()
    icir = ic_mean / ic_std
    t_ic = icir * np.sqrt(len(ics))
    ls_ann = (1 + ls).prod() ** (12 / len(ls)) - 1
    rows.append(dict(因子=name, IC=ic_mean, ICIR=icir, t=t_ic, IC负比例=(ics < 0).mean(),
                     多空年化=ls_ann, 单调性=np.nanmean(monot), N月=len(ics)))

R = pd.DataFrame(rows)
R['|ICIR|'] = R['ICIR'].abs()
R = R.sort_values('|ICIR|', ascending=False)

# BH-FDR
from scipy import stats
R['p'] = 2 * (1 - stats.norm.cdf(R['t'].abs()))
srt = R.sort_values('p')
m = len(srt)
srt['p_adj'] = np.minimum.accumulate((srt['p'] * m / np.arange(1, m + 1))[::-1])[::-1]
R = R.merge(srt[['因子', 'p_adj']], on='因子', how='left')
R['FDR5%'] = np.where(R['p_adj'] < 0.05, '✅', '')

print('\n=== 18 个因子月度截面检验（%d 个月，%s ~ %s）===' % (R['N月'].min(), me[0].date(), me[-1].date()))
print('%-12s %8s %8s %7s %8s %8s %8s %6s' % ('因子', 'IC', 'ICIR', 't', '多空年化', '单调性', 'p_adj', 'FDR'))
for _, r in R.iterrows():
    print('%-12s %8.4f %8.3f %7.2f %7.1f%% %8.2f %8.3f %6s' % (
        r['因子'], r['IC'], r['ICIR'], r['t'], r['多空年化'] * 100, r['单调性'], r['p_adj'], r['FDR5%']))

R.to_csv('factor_lab_results.csv', index=False)
print('\n已保存 factor_lab_results.csv')
