# -*- coding: utf-8 -*-
"""Lockbox 一次性验证：只能运行一次，评估封存段（训练/调参全程未接触）

严格性：
  - 标签在 data/big/lockbox_labels.npz，主数组 arr.npz 中该段 Y/Y2 已置 NaN
  - 模型权重与 Stacking 系数全部来自重训后的 checkpoint，不再拟合任何参数
  - 运行次数写入 reports/lockbox_state.json，第二次默认拒绝
用法：python scripts/lockbox_eval.py [--force]
"""
import os, sys, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
import lightgbm as lgb
from scipy import stats as sps

ROOT = r'D:/quant'
FORCE = '--force' in sys.argv
DRY = '--dry-run' in sys.argv          # 只验证流程，不写 state / 不消耗一次性机会
STATE = f'{ROOT}/reports/lockbox_state.json'
if os.path.exists(STATE):
    st = json.load(open(STATE))
    if st.get('runs', 0) >= 1 and not FORCE:
        print('拒绝执行：lockbox 已于 %s 评估过（runs=%d）。' %
              (st.get('last_run'), st['runs']))
        print('一次性验证的意义就在于不再窥视；如需复核请查看 reports/lockbox_result.json')
        sys.exit(2)

L, D, HEADS, LAYERS, DROP = 40, 96, 4, 2, 0.15
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- 与 big_train.py 完全一致的 TFNet ----
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class SwiGLU(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__(); h = int(d * mult)
        self.w1 = nn.Linear(d, h); self.w3 = nn.Linear(d, h); self.w2 = nn.Linear(h, d)
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        return self.drop(self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1 = RMSNorm(d); self.att = nn.MultiheadAttention(d, h, dropout=DROP, batch_first=True)
        self.n2 = RMSNorm(d); self.ff = SwiGLU(d); self.dp = nn.Dropout(DROP)
    def forward(self, x):
        h = self.n1(x); a, _ = self.att(h, h, h, need_weights=False)
        x = x + self.dp(a)
        return x + self.ff(self.n2(x))

class TFNet(nn.Module):
    def __init__(self, f, d, h, nl):
        super().__init__()
        self.emb = nn.Linear(f, d); self.pos = nn.Parameter(torch.zeros(1, L, d))
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(nl)])
        self.norm = RMSNorm(d); self.head = nn.Linear(d * 2, 1); self.drop = nn.Dropout(DROP)
    def forward(self, x):
        z = self.drop(self.emb(x) + self.pos)
        for b in self.blocks: z = b(z)
        z = self.norm(z)
        return self.head(torch.cat([z[:, -1], z.mean(1)], -1)).squeeze(-1)

z = np.load(f'{ROOT}/data/big/arr.npz', allow_pickle=True)
X, M = z['X'], z['M']
dates = pd.to_datetime(z['dates']); FEATS = list(z['feats']); T, N, F = X.shape
lk = np.load(f'{ROOT}/data/big/lockbox_labels.npz')
ti, ni, y = lk['ti'], lk['ni'], lk['y2']
meta = json.load(open(f'{ROOT}/reports/big_models_meta.json'))
coef = np.array(meta['coef'])
order = meta.get('feature_order', FEATS)
wgt = np.array([meta['weights'].get(str(f), 0.0) for f in order])
lmeta = json.load(open(f'{ROOT}/checkpoints/lgbm_meta.json'))
print('封存段 %s 起 | %d 行 / %d 日 | 系数 %s' %
      (str(lk['lock_start']), len(ti), len(np.unique(ti)), np.round(coef, 3)), flush=True)

def make_seq(tt, ii):
    idx = tt[:, None] - np.arange(L - 1, -1, -1)[None, :]
    return X[idx, ii[:, None], :]

# 1) LightGBM
t0 = time.time()
booster = lgb.Booster(model_file=f'{ROOT}/checkpoints/lgbm_big.txt')
p_lgb = booster.predict(X[ti, ni, :], num_iteration=lmeta['best_iter'])
print('LGBM %.0fs' % (time.time() - t0), flush=True)

# 2) TF 3 seed bagging
p_tf = np.zeros(len(ti), dtype=np.float64)
for s in (1, 2, 3):
    m = TFNet(F, D, HEADS, LAYERS).to(DEV)
    m.load_state_dict(torch.load(f'{ROOT}/checkpoints/tf_seed{s}.pt',
                                 map_location=DEV, weights_only=True))
    m.eval(); out = np.empty(len(ti), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(ti), 4096):
            b = torch.from_numpy(make_seq(ti[a:a + 4096], ni[a:a + 4096])).to(DEV)
            out[a:a + 4096] = m(b).cpu().numpy()
    p_tf += out / 3
    print('TF seed%d %.0fs' % (s, time.time() - t0), flush=True)

# 3) 因子合成（逐日截面 rank @ ICIR 权重）
fac = np.zeros(len(ti), dtype=np.float64)
pos = pd.Series(np.arange(len(ti))).groupby(ti).groups
for t, rows in pos.items():
    ii_all = np.where(M[t])[0]
    r = np.full((N, F), np.nan, np.float32)
    sub = X[t, ii_all, :]
    r[ii_all] = ((sub.argsort(0).argsort(0) + .5) / len(ii_all)).astype(np.float32)
    f_t = np.nan_to_num(r) @ wgt
    ii_rows = np.array(rows)
    fac[ii_rows] = f_t[ni[ii_rows]]

def zc(a):
    a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-12)

# 标准化参数：优先用训练时冻结在 meta 段的 mu/sd（Ridge 口径，见 stack_ridge.py）；
# 老 checkpoint 没有这两个字段时回退到逐段 zc（OLD/OLS 口径），保持可复现。
_meta = meta
if 'coef_mu' in _meta and 'coef_sd' in _meta:
    mu_, sd_ = np.array(_meta['coef_mu']), np.array(_meta['coef_sd'])
    A = np.column_stack([(p_lgb - mu_[0]) / sd_[0], (p_tf - mu_[1]) / sd_[1],
                         (fac - mu_[2]) / sd_[2], np.ones(len(ti))])
    print('Stacking 标准化：固定 meta 段 mu/sd（Ridge 口径）', flush=True)
else:
    A = np.column_stack([zc(p_lgb), zc(p_tf), zc(fac), np.ones(len(ti))])
    print('Stacking 标准化：逐段 zc（旧 OLS 口径回退）', flush=True)
p_st = A @ coef

# ---- 评估 ----
def ic(p):
    d = pd.DataFrame({'d': ti, 'p': p, 'y': y})
    s = d.groupby('d').apply(
        lambda g: sps.spearmanr(g['p'], g['y']).correlation if len(g) > 30 else np.nan).dropna()
    return s

def portfolio(p, q=.1, cost=.002):
    d = pd.DataFrame({'t': ti, 'i': ni, 'p': p, 'y': y})
    rec = []
    for t, g in d.groupby('t'):
        if len(g) < 100: continue
        n = max(int(len(g) * q), 10); gs = g.sort_values('p')
        top = gs.tail(n)['y'].mean() - g['y'].mean()
        bot = gs.head(n)['y'].mean() - g['y'].mean()
        rest = g[~g.index.isin(gs.head(n).index)]['y'].mean() - g['y'].mean()
        rec.append((dates[t], top - bot, top, rest))
    r = pd.DataFrame(rec, columns=['d', 'ls', 'top', 'exbot']).set_index('d').iloc[::5]
    out = {}
    for c, fee in [('ls', cost * 4), ('top', cost * .5), ('exbot', cost * .1)]:
        v = r[c] - fee
        ann = (1 + v).prod() ** (50.4 / len(v)) - 1; vol = v.std() * np.sqrt(50.4)
        out[c] = dict(ann=float(ann), vol=float(vol), sharpe=float((ann - .02) / vol))
    return out

res = {'runs_at': time.strftime('%Y-%m-%d %H:%M:%S'),
       'lock_start': str(lk['lock_start']), 'n_rows': int(len(ti)),
       'n_days': int(len(np.unique(ti))), 'coef': coef.tolist(),
       'models': {}, 'note': '一次性验证；模型与系数均冻结，未对封存段做任何拟合'}
for nm, p in [('因子合成', fac), ('LGBM', p_lgb), ('TF', p_tf), ('Stacking', p_st)]:
    s = ic(p)
    res['models'][nm] = dict(IC=float(s.mean()), ICIR=float(s.mean() / s.std()),
                             portfolio=portfolio(p))
    print('%-9s IC %+.4f ICIR %+.2f | 负向筛选年化 %+.2f%%' %
          (nm, s.mean(), s.mean() / s.std(),
           res['models'][nm]['portfolio']['exbot']['ann'] * 100), flush=True)

s = ic(p_st)
yr_arr = dates[np.array(s.index)].year
res['by_year'] = {str(yr): float(s[yr_arr == yr].mean()) for yr in sorted(set(yr_arr))}

# ---- 最终降频配置在封存段的表现（配置只在开发段选出，此处不再调参）----
fr = pd.read_csv(f'{ROOT}/reports/turnover_frontier.csv')
cand = fr[(fr['mode'] == 'exbot') & (fr['turnover'] < .10)]
if len(cand):
    cfg = cand.loc[cand.sharpe.idxmax()]
    STEP, BUF = int(cfg.step), float(cfg.buffer)
else:
    STEP, BUF = 5, 0.0
Ylock = np.full((T, N), np.nan, np.float32)
Ylock[ti, ni] = y.astype(np.float32)

def lowfreq(p, step, buf, q=.1, cost=.004):
    d = pd.DataFrame({'t': ti, 'i': ni, 'p': p, 'y': y})
    GROUPS = {t: g.sort_values('p', ascending=False).reset_index(drop=True)
              for t, g in d.groupby('t')}
    all_t = np.sort(d['t'].unique()); reb = all_t[::step][:-1]
    rec, turns, prev = [], [], set()
    for t0 in reb:
        g = GROUPS.get(t0)
        if g is None or len(g) < 100: continue
        n = max(int(len(g) * q), 10)
        pct = 1 - np.arange(len(g)) / len(g); rank = dict(zip(g['i'].values, pct))
        keep_bot = {i for i in prev if rank.get(i, 0) <= q + buf} if prev else set()
        bot = set(list(keep_bot)[:n])
        need = n - len(bot)
        if need > 0:
            bot |= set([i for i in g['i'].values[-n:] if i not in bot][:need])
        turn = (len(bot - prev) / max(len(bot), 1) if prev else 1.0) * q
        turns.append(turn); prev = bot
        rest = g.loc[~g['i'].isin(bot), 'i'].values.astype(int)
        rs = [Ylock[min(t0 + 5 * k, T - 1), rest] for k in range(step // 5)]
        rec.append(float(np.nanmean(np.nansum(np.vstack(rs), 0))))
    net = np.array(rec) - np.array(turns) * cost
    periods = 252.0 / step
    ann = (1 + net).prod() ** (periods / len(net)) - 1
    vol = net.std() * np.sqrt(periods)
    return dict(step=step, buffer=buf, turnover=float(np.mean(turns)),
                ann=float(ann), vol=float(vol),
                sharpe=float((ann - .02) / vol) if vol > 0 else None, n=len(net))

res['lowfreq'] = lowfreq(p_st, STEP, BUF)
print('封存段最终配置：周期 %d 日 / buffer %.1f → 换手 %.1f%% | 年化 %+.2f%% | 夏普 %.2f' %
      (STEP, BUF, res['lowfreq']['turnover'] * 100,
       res['lowfreq']['ann'] * 100, res['lowfreq']['sharpe']), flush=True)
res['forced'] = FORCE
if DRY:
    print('\n[DRY RUN] 流程验证完成，未写 lockbox_result.json / 未消耗一次性机会')
else:
    json.dump(res, open(f'{ROOT}/reports/lockbox_result.json', 'w'), indent=2, ensure_ascii=False)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {'runs': 0}
    st['runs'] = st.get('runs', 0) + 1; st['last_run'] = res['runs_at']
    st.setdefault('history', []).append({'at': res['runs_at'], 'forced': FORCE,
                                         'stacking_IC': res['models']['Stacking']['IC']})
    json.dump(st, open(STATE, 'w'), indent=2)
    print('\n已存 reports/lockbox_result.json | state runs=%d' % st['runs'])
