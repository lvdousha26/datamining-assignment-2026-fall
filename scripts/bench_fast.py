# -*- coding: utf-8 -*-
"""基线对照（快的那三个）：TF-IDF+LR / LightGBM（数值表格）/ LightGBM（数值+文本特征）

目的不是刷分，而是回答一个问题：
**「文本」到底有没有增量？** 如果只给公告里的数字（净利润变动幅度、金额、报告期），
LightGBM 就能拿到接近甚至超过文本模型的分数，那文本模型的价值就得重新论证。

用法：python scripts/bench_fast.py
产出：runs/laya_yjyg/bench_fast.json（准确率 + 单条延迟）
"""
import json
import os
import sys
import time
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
DIR = f'{ROOT}/runs/laya_yjyg'
LAB = ['利好', '中性', '利空']


def load_meta():
    """把 train/eval 的 text 映射回原始公告的数值字段（不重新抽样，保证切分一致）"""
    fs = sorted(glob.glob(f'{ROOT}/data/big/yjyg/*.parquet'))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d['text'] = (d['预测指标'].fillna('') + '｜' + d['业绩变动'].fillna('')).str.strip('｜')
    d['text'] = d['text'].str.replace(r'\s+', ' ', regex=True)
    # 同一 text 可能重复（不同公司同模板），只取第一次出现
    d = d.drop_duplicates(subset=['text'], keep='first')
    d['rpt'] = pd.to_datetime(d['公告日期'], errors='coerce')
    d['quarter'] = d['rpt'].dt.month.map({3: 1, 6: 2, 9: 3, 12: 4})
    d['amp'] = pd.to_numeric(d['业绩变动幅度'], errors='coerce')
    d['fc'] = pd.to_numeric(d['预测数值'], errors='coerce')
    d['ly'] = pd.to_numeric(d['上年同期值'], errors='coerce')
    d['fc_sign'] = np.sign(d['fc'].fillna(0))
    d['ly_sign'] = np.sign(d['ly'].fillna(0))
    d['log_ly'] = np.log1p(d['ly'].abs())
    d['log_fc'] = np.log1p(d['fc'].abs())
    d['txt_len'] = d['text'].str.len()
    d['has_pct'] = d['text'].str.contains('%|万元|亿元').astype(int)
    d['ind'] = d['预测指标'].astype('category').cat.codes
    return d[['text', 'quarter', 'amp', 'fc_sign', 'ly_sign', 'log_ly', 'log_fc',
              'txt_len', 'has_pct', 'ind']]


TAB_COLS = ['quarter', 'amp', 'fc_sign', 'ly_sign', 'log_ly', 'log_fc', 'txt_len', 'has_pct', 'ind']


def text_feats(tr_txt, ev_txt):
    """轻量文本特征：字符 n-gram 计数（不建 vocab，直接哈希），供 LightGBM 用"""
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(analyzer='char_wb', ngram_range=(2, 3), n_features=256, alternate_sign=False)
    return hv.transform(tr_txt), hv.transform(ev_txt)


def timeit(fn, texts, n=200):
    t0 = time.time()
    for x in texts[:n]:
        fn(x)
    return (time.time() - t0) / min(n, len(texts)) * 1000


def main():
    tr = pd.read_csv(f'{DIR}/train.csv')
    ev = pd.read_csv(f'{DIR}/eval.csv')
    meta = load_meta()
    tr = tr.merge(meta, on='text', how='left')
    ev = ev.merge(meta, on='text', how='left')
    print('train %d / eval %d | 表格列缺失率: %s' % (
        len(tr), len(ev), tr[TAB_COLS].isna().mean().round(3).to_dict()))
    print('标签分布 train %s | eval %s' % (tr.label.value_counts().to_dict(), ev.label.value_counts().to_dict()))

    out = {}
    # ---------------- 1. TF-IDF + LR ----------------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tv = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), min_df=2, max_features=20000)
    Xtr, Xev = tv.fit_transform(tr.text), tv.transform(ev.text)
    lr = LogisticRegression(max_iter=1000, C=4.0, class_weight='balanced', n_jobs=-1)
    t0 = time.time(); lr.fit(Xtr, tr.label); dur = time.time() - t0
    p = lr.predict(Xev)
    acc = (p == ev.label.values).mean()
    # 单条推理延迟（向量化+预测）
    lat = timeit(lambda x: lr.predict(tv.transform([x]))[0], ev.text.tolist())
    out['lr_tfidf'] = {'acc': float(acc), 'train_s': dur, 'ms': lat, 'n_feat': int(Xtr.shape[1])}
    print('\n[TF-IDF+LR] acc %.4f | 训练 %.1fs | 特征 %d | 推理 %.2f ms/条' % (acc, dur, Xtr.shape[1], lat))

    # ---------------- 2. LightGBM（纯数值表格）----------------
    import lightgbm as lgb
    dtr = tr[TAB_COLS].astype(float)
    dev = ev[TAB_COLS].astype(float)
    m1 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                            class_weight='balanced', verbose=-1)
    t0 = time.time(); m1.fit(dtr.values, tr.label); dur1 = time.time() - t0
    p1 = m1.predict(dev.values)
    acc1 = (p1 == ev.label.values).mean()
    lat1 = timeit(lambda x: m1.predict(np.asarray(x, dtype=float).reshape(1, -1))[0],
                  dev.values.tolist())
    out['lgbm_tabular'] = {'acc': float(acc1), 'train_s': dur1, 'ms': lat1}
    print('[LightGBM 纯表格] acc %.4f | 训练 %.1fs | 推理 %.2f ms/条' % (acc1, dur1, lat1))

    # ---------------- 3. LightGBM（表格 + 文本哈希特征）----------------
    Str, Sev = text_feats(tr.text.tolist(), ev.text.tolist())
    from scipy.sparse import hstack
    A = hstack([dtr.values.astype(np.float32), Str]).tocsr()
    B = hstack([dev.values.astype(np.float32), Sev]).tocsr()
    m2 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                            class_weight='balanced', verbose=-1)
    t0 = time.time(); m2.fit(A, tr.label); dur2 = time.time() - t0
    p2 = m2.predict(B)
    acc2 = (p2 == ev.label.values).mean()
    out['lgbm_tab_text'] = {'acc': float(acc2), 'train_s': dur2, 'ms': 1.0}
    print('[LightGBM 表格+文本] acc %.4f | 训练 %.1fs' % (acc2, dur2))

    # ---------------- 4. 关键词规则（对照基准）----------------
    def rule(t):
        t = str(t)
        if any(k in t for k in ['扭亏', '减亏']):
            return '利好'
        if any(k in t for k in ['减少', '下降', '亏损', '预减', '转亏', '增亏']):
            return '利空'
        if any(k in t for k in ['增长', '增加', '上升', '盈利', '续盈']):
            return '利好'
        return '中性'
    acc_r = (ev.text.map(rule) == ev.label).mean()
    lat_r = timeit(rule, ev.text.tolist())
    out['regex'] = {'acc': float(acc_r), 'ms': lat_r}
    print('[关键词规则] acc %.4f | 推理 %.3f ms/条' % (acc_r, lat_r))

    maj = ev.label.value_counts(normalize=True).max()
    print('\n多数类基线 %.4f' % maj)
    json.dump({'majority': float(maj), **out}, open(f'{DIR}/bench_fast.json', 'w'),
              ensure_ascii=False, indent=1)
    print('→', f'{DIR}/bench_fast.json')


if __name__ == '__main__':
    with RunLogger(__file__, tag='bench_fast'):
        main()
