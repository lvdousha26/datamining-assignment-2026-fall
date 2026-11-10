# -*- coding: utf-8 -*-
"""Laya 在业绩预告上的准确率（零样本 vs 微调后，同一套 eval 样本）

用法：
  python scripts/laya_yjyg_eval.py                              # 零样本基线
  python scripts/laya_yjyg_eval.py --model runs/laya_yjyg/ft \  # 微调后
                                   --tag ft
产出：runs/laya_yjyg/eval_<tag>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
DIR = f'{ROOT}/runs/laya_yjyg'
# 与 laya-train 内部合成的问法完全一致，否则 before/after 不可比
QID = 'label'
QS = {QID: {"type": "choice",
            "instructions": "Classify the input into the correct category.",
            "criteria": ['利好', '中性', '利空']}}

LABELS = ['利好', '中性', '利空']


def main(model, tag, limit=0):
    import laya
    ev = pd.read_csv(f'{DIR}/eval.csv')
    if limit:
        ev = ev.head(limit)
    t0 = time.time()
    kw = dict(subfolder='multilingual') if model == 'convaiinnovations/laya' else {}
    ag = laya.Agent(model_id_or_path=model, device='cuda', **kw)
    print('加载 %s（%.1fs）| eval %d 行' % (model, time.time() - t0, len(ev)), flush=True)

    pred, probs, t1 = [], [], time.time()
    for i, x in enumerate(ev['text'], 1):
        r = ag.predict(x, QS, lang='zh')
        a = r['answers'][QID]
        pred.append(a['choice'])
        probs.append([a['probabilities'].get(k, 0) for k in LABELS])
        if i % 100 == 0:
            print('  %d/%d  已用 %.0fs' % (i, len(ev), time.time() - t1), flush=True)

    y = ev['label'].tolist()
    acc = np.mean([p == t for p, t in zip(pred, y)])
    base = max(pd.Series(y).value_counts(normalize=True))   # 全猜最多的类
    cm = pd.crosstab(pd.Series(y, name='真实'), pd.Series(pred, name='预测')).reindex(
        index=LABELS, columns=LABELS, fill_value=0)
    rec = {k: (cm.loc[k, k] / cm.loc[k].sum() if cm.loc[k].sum() else np.nan) for k in LABELS}
    prec = {k: (cm.loc[k, k] / cm[k].sum() if cm[k].sum() else np.nan) for k in LABELS}
    conf = np.array(probs)
    print('\n=== %s ===' % tag)
    print('accuracy %.4f | 多数类基线 %.4f | 提升 %+.2fpp' % (acc, base, (acc - base) * 100))
    print('平均置信度 %.3f | 用时 %.0fs（%.0fms/条）' % (
        conf.max(1).mean(), time.time() - t1, (time.time() - t1) / len(ev) * 1000))
    print('\n混淆矩阵（行=真实，列=预测）')
    print(cm.to_string())
    print('\n每类召回：' + ' | '.join('%s %.3f' % (k, rec[k]) for k in LABELS))
    print('每类精确：' + ' | '.join('%s %.3f' % (k, prec[k]) for k in LABELS))

    pd.DataFrame({'text': ev['text'], 'label': y, 'pred': pred,
                  'p_pos': conf[:, 0], 'p_neu': conf[:, 1], 'p_neg': conf[:, 2]}
                 ).to_csv(f'{DIR}/pred_{tag}.csv', index=False)
    json.dump({'tag': tag, 'model': model, 'n': len(ev), 'acc': float(acc),
               'majority_baseline': float(base), 'recall': rec, 'precision': prec,
               'mean_conf': float(conf.max(1).mean()),
               'ms_per_item': (time.time() - t1) / len(ev) * 1000,
               'confusion': cm.values.tolist(), 'labels': LABELS},
              open(f'{DIR}/eval_{tag}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('→', f'{DIR}/eval_{tag}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='convaiinnovations/laya')
    ap.add_argument('--subfolder', default='multilingual')
    ap.add_argument('--tag', default='zeroshot')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    m = a.model if 'runs' in a.model or os.path.isabs(a.model) else a.model
    with RunLogger(__file__, tag=a.tag):
        main(m, a.tag, a.limit)
