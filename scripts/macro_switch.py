# -*- coding: utf-8 -*-
"""独立宏观开关：信用利差 + 股债相关性
逻辑：
  信用利差走阔 = 信用风险上升 → 减风险资产
  股债相关性转正 = 传统对冲失效（通胀/流动性冲击型） → 减风险资产
两者都与"价格趋势"独立，是策略③内部信号之外的新信息
全部信号滞后一个月，无前视
"""
import io, contextlib, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'C:/Users/25097/WorkBuddy/2026-10-30-23-31-35'
sys.path.insert(0, BASE); os.chdir(BASE)
with contextlib.redirect_stdout(io.StringIO()):
    import fx_compare as F, spec3

def pm(px):
    r = (1 + px.pct_change()).resample('ME').prod() - 1
    r.index = r.index.to_period('M'); return r

# ---- 信用利差 ----
B = pd.read_pickle(r'D:/quant/bond_5y.pkl')
bm = B.resample('ME').last(); bm.index = bm.index.to_period('M')
spread = (bm['中短期票据(AAA)'] - bm['国债']).dropna()      # 单位 %
z = ((spread - spread.rolling(36).mean()) / spread.rolling(36).std()).clip(-2, 3)

# ---- 股债相关性 ----
R = pd.DataFrame({'股': pm(pd.read_pickle('sh_index.pkl')['close']),
                  '债': pm(pd.read_pickle('idx_000012.pkl'))}).dropna()
corr = R['股'].rolling(24).corr(R['债'])

# ---- 广度（对照，已知的那个）----
RISK = {'沪深300': pm(pd.read_pickle('sh_index.pkl')['close']),
        '中证500': pm(pd.read_pickle('idx_000905.pkl')),
        '纳指': pm(F.NDX_R), '红利': F.legs['div'], 'BRK': pm(F.BRK_R)}
PX = {k: (1+v).cumprod() for k, v in RISK.items()}
BREADTH = pd.DataFrame({k: (PX[k] > PX[k].rolling(10).mean()).astype(float) for k in PX}).mean(axis=1)

s3 = spec3.s3B.loc['2011-01':].dropna()
idx = s3.index
z1 = z.reindex(idx).ffill().shift(1)
c1 = corr.reindex(idx).ffill().shift(1)
b1 = BREADTH.reindex(idx).ffill().shift(1)

def st(r, tag):
    r = r.dropna(); a = (1+r).prod()**(12/len(r))-1; v = r.std()*12**.5
    nav = (1+r).cumprod(); dd = (nav/nav.cummax()-1).min()
    print('  %-34s 年化 %6.2f%% 波动 %5.2f%% 夏普 %5.2f 回撤 %6.1f%% Calmar %.2f' % (
        tag, a*100, v*100, (a-.02)/v, dd*100, abs(a/dd)))
    return a, (a-.02)/v, dd

print('样本 %s ~ %s (%d月)  [信用利差数据 2010 起]' % (idx[0], idx[-1], len(idx)))
print('\n=== A. 单信号 ===')
st(s3, '策略③ 原版')
st(s3*(1 - 0.5*np.clip(z1, 0, 2)/2), '信用利差高 → 最多减半(连续)')
st(s3*(1 - 0.5*(z1 > 1).astype(float)), '信用利差 z>1 → 减半(阈值)')
st(s3*(1 - 0.5*(c1 > 0).astype(float)), '股债相关>0 → 减半')
st(s3*(1 - 0.5*(c1 > 0.3).astype(float)), '股债相关>0.3 → 减半')
st(s3*(1 - 0.5*(b1 < 0.4).astype(float)), '广度<40% → 减半(对照)')

print('\n=== B. 组合 ===')
m1 = (1 - 0.4*(z1 > 0.8) - 0.4*(c1 > 0)).clip(0.3, 1)
st(s3*m1, '信用利差 + 股债相关 (各减40%)')
m2 = (1 - 0.25*(z1 > 0.8) - 0.25*(c1 > 0) - 0.25*(b1 < 0.4)).clip(0.25, 1)
st(s3*m2, '三者合计(各减25%)')
m3 = (1 - 0.3*(z1 > 0.8) - 0.3*(c1 > 0) - 0.2*(b1 < 0.4)).clip(0.2, 1)
st(s3*m3, '三者加权(0.3/0.3/0.2)')

print('\n=== C. 分段检验（防单窗口）===')
for a, b, lab in [('2011-01', '2018-12', '2011-2018'), ('2019-01', '2026-10', '2019-2026')]:
    print(' 【%s】' % lab)
    st(s3.loc[a:b], '  原版')
    st((s3*(1 - 0.5*(z1 > 1).astype(float))).loc[a:b], '  信用利差z>1减半')
    st((s3*(1 - 0.5*(c1 > 0).astype(float))).loc[a:b], '  股债相关>0减半')
    st((s3*m3).loc[a:b], '  三者加权')

print('\n=== D. 信号本身有没有预测力 ===')
fwd = R['股'].shift(-1)
for nm, sig in [('信用利差z', z), ('股债相关', corr)]:
    s = sig.reindex(fwd.index).dropna()
    common = s.index.intersection(fwd.dropna().index)
    print('  %s vs 次月沪深300: 相关 %+.2f | 信号高分位次月均值 %+.2f%% vs 低分位 %+.2f%%' % (
        nm, s.loc[common].corr(fwd.loc[common]),
        fwd.loc[common][s.loc[common] > s.loc[common].quantile(0.7)].mean()*100,
        fwd.loc[common][s.loc[common] < s.loc[common].quantile(0.3)].mean()*100))

print('\n=== 当前读数 ===')
print('  信用利差(AAA-国债5Y) %.0fbp | z=%+.2f（%.0f分位）' % (
    spread.iloc[-1]*100, z.iloc[-1], (z < z.iloc[-1]).mean()*100))
print('  股债相关性(24月) %+.2f' % corr.iloc[-1])
print('  广度 %.0f%%' % (BREADTH.reindex(idx).ffill().iloc[-1]*100))

fig, ax = plt.subplots(3, 1, figsize=(11, 9), height_ratios=[2.4, 1, 1], sharex=True)
nav = (1+s3).cumprod(); nav2 = (1+s3*m3).cumprod()
ax[0].plot(nav.index.to_timestamp('M'), nav.values, color='#7f8c8d', lw=1.2, label='策略③ 原版')
ax[0].plot(nav2.index.to_timestamp('M'), nav2.values, color='#c0392b', lw=1.8, label='+ 宏观开关')
ax[0].set_yscale('log'); ax[0].legend(frameon=False); ax[0].grid(alpha=.25, which='both')
ax[0].set_title('信用利差 + 股债相关性 作为风险开关（无前视）', fontsize=11)
ax[1].plot(spread.index.to_timestamp('M'), spread.values*100, color='#2980b9', lw=1.1)
ax[1].set_ylabel('信用利差 bp'); ax[1].grid(alpha=.25)
ax[2].plot(corr.index.to_timestamp('M'), corr.values, color='#8e44ad', lw=1.1)
ax[2].axhline(0, color='#7f8c8d', ls=':', lw=1)
ax[2].set_ylabel('股债相关'); ax[2].grid(alpha=.25)
plt.tight_layout(); plt.savefig(r'D:/quant/reports/macro_switch.png', dpi=135, bbox_inches='tight')
print('\n图已保存 D:/quant/reports/macro_switch.png')
