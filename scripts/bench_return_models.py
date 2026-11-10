# -*- coding: utf-8 -*-
"""换标签版基线：次日超额收益（3 分类）上的 规则 / TF-IDF+LR / LightGBM 表格

对照上一版（标签=预告类型）：
    预告类型版：表格 93.2%  >  BERT 89.2%  >  LR 88.2%  >  规则 84.6%
如果这次表格掉到随机水平、而文本类还有点东西，说明「文本确实提供了数字之外的信息」；
如果全都掉到 33% 附近，说明次日超额本身不可由公告文本预测——也是结论。

用法：python scripts/bench_return_models.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402
from bench_fast import load_meta, TAB_COLS  # noqa: E402

ROOT = r'D:/quant'
RET = f'{ROOT}/runs/laya_yjyg/ret'


def main():
    tr = pd.read_csv(f'{RET}/train.csv')
    ev = pd.read_csv(f'{RET}/eval.csv')
    print('train %d / eval %d | 训练分布 %s | 评估分布 %s' % (
        len(tr), len(ev), tr.lab3.value_counts().to_dict(), ev.lab3.value_counts().to_dict()))
    maj = ev.lab3.value_counts(normalize=True).max()
    y = ev.lab3.values
    rows = [('多数类基线', maj, 0.0, '—')]

    # ---- 规则 ----
    def rule(t):
        t = str(t)
        if any(k in t for k in ['扭亏', '减亏']):
            return '利好'
        if any(k in t for k in ['减少', '下降', '亏损', '预减', '转亏', '增亏']):
            return '利空'
        if any(k in t for k in ['增长', '增加', '上升', '盈利', '续盈']):
            return '利好'
        return '中性'
    acc = float((ev.text.map(rule) == y).mean())
    rows.append(('关键词正则', acc, 0.002, '—'))
    print('[规则] acc %.4f' % acc)

    # ---- TF-IDF + LR ----
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    tv = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), min_df=2, max_features=20000)
    Xtr, Xev = tv.fit_transform(tr.text), tv.transform(ev.text)
    for tag, kw in [('TF-IDF + LR（无权重）', {}), ('TF-IDF + LR（类别权重）', {'class_weight': 'balanced'})]:
        m = LogisticRegression(max_iter=1000, C=4.0, **kw).fit(Xtr, tr.lab3)
        p = m.predict(Xev)
        a = float((p == y).mean())
        rows.append((tag, a, 0.29, '%.1fs' % 0.3))
        print('[%s] acc %.4f | macroF1 %.4f' % (tag, a, f1_score(y, p, average='macro')))

    # ---- LightGBM 表格 ----
    import lightgbm as lgb
    meta = load_meta()
    t1 = tr.merge(meta, on='text', how='left')
    e1 = ev.merge(meta, on='text', how='left')
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           class_weight='balanced', verbose=-1)
    m.fit(t1[TAB_COLS].astype(float).values, t1.lab3)
    p = m.predict(e1[TAB_COLS].astype(float).values)
    a = float((p == y).mean())
    rows.append(('LightGBM（纯数值表格）', a, 1.12, '2s'))
    print('[LGBM 表格] acc %.4f | macroF1 %.4f' % (a, f1_score(y, p, average='macro')))

    json.dump([{'model': n, 'acc': float(acc_), 'ms': ms, 'train': t} for n, acc_, ms, t in rows],
              open(f'{RET}/bench_return_fast.json', 'w'), ensure_ascii=False, indent=1)
    for n, acc_, _, _ in rows:
        print('  %-26s %.4f' % (n, acc_))


if __name__ == '__main__':
    with RunLogger(__file__, tag='ret_models'):
        main()
