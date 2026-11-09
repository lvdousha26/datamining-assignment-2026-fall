# -*- coding: utf-8 -*-
"""对照：AdamW 是否只是 lr 没调好？
用与 Phase A 完全相同的设置（seed=1 / 3ep / sub=0.25）扫 AdamW lr，
避免把「Muon 赢」误判成「AdamW 基线太弱」。
"""
import pandas as pd
import muon_ab as M

M.ROOT_ROWS = []; M.CURVES = {}; M.PH = 'C'
LRS = [4e-4, 8e-4, 1.6e-3, 3e-3]
rows = []
print('\n=== AdamW lr 对照扫描（与 Phase A 同设置）===', flush=True)
for lr in LRS:
    rows.append(M.run('AdamW_%g' % lr, 'adam', lr, 1, 3, .25))

d = pd.DataFrame(rows)
d.to_csv(f'{M.ROOT}/reports/adamw_lr_control.csv', index=False)
print('\n=== 结果 ===', flush=True)
print(d[['tag', 'lr', 'ic22', 'ic23', 'ic24', 'loss_final']].round(4).to_string(index=False))
a = d.loc[d.ic22.idxmax()]
print('\nAdamW 最佳 lr=%g → 2022 IC %+.4f | 2024+ %+.4f' % (a.lr, a.ic22, a.ic24))
print('已存 reports/adamw_lr_control.csv')
