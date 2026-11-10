# -*- coding: utf-8 -*-
"""Laya 微调数据准备：业绩预告 → (text, 利好/利空/中性)

标签来源
--------
交易所强制披露的「预告类型」，不需要人工标注：
  利好 = 预增 / 略增 / 扭亏 / 续盈 / 减亏
  利空 = 预减 / 略减 / 首亏 / 续亏 / 增亏
  中性 = 不确定

文本
----
`预测指标 + 业绩变动` 拼成一句，例如
  「归属于上市公司股东的净利润｜预计2016年1-3月…净利润盈利:4,878万元-5,318.89万元, 比上年同期减少0%-10%」

**必须说清楚的偏差**：正文里通常直接写着「增长 120%」「减少 0%-10%」，
所以这个任务本质是「读懂公告里的数字方向」，不是纯语义情感。好处是标签干净、零标注成本；
坏处是准确率会偏高，别拿它当"模型懂语义"的证据（报告里已注明）。

用法：python scripts/laya_yjyg_prep.py --n 3000
产出：runs/laya_yjyg/{train.csv, eval.csv, meta.json}
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
OUT = f'{ROOT}/runs/laya_yjyg'
POS = {'预增', '略增', '扭亏', '续盈', '减亏'}
NEG = {'预减', '略减', '首亏', '续亏', '增亏'}
NEU = {'不确定'}


def main(n=3000, seed=0):
    os.makedirs(OUT, exist_ok=True)
    fs = sorted(glob.glob(f'{ROOT}/data/big/yjyg/*.parquet'))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    print('原始 %d 行 / %d 只 / %d 期' % (len(d), d['股票代码'].nunique(), len(fs)))

    def lab(t):
        if t in POS:
            return '利好'
        if t in NEG:
            return '利空'
        if t in NEU:
            return '中性'
        return None

    d['label'] = d['预告类型'].map(lab)
    d = d.dropna(subset=['label'])
    d['text'] = (d['预测指标'].fillna('') + '｜' + d['业绩变动'].fillna('')).str.strip('｜')
    d['text'] = d['text'].str.replace(r'\s+', ' ', regex=True)
    d['len'] = d['text'].str.len()
    d = d[(d['len'] >= 15) & (d['len'] <= 200)]
    d = d.drop_duplicates(subset=['股票代码', '公告日期', '预测指标', '业绩变动'])
    print('清洗后 %d 行 | 分布 %s' % (len(d), d['label'].value_counts().to_dict()))

    # 分层抽样：按原始比例，不人为平衡（否则 eval 分布与实盘不符）
    frac = d['label'].value_counts(normalize=True)
    parts = []
    for lab_, p in frac.items():
        k = int(round(n * p))
        parts.append(d[d['label'] == lab_].sample(min(k, (d['label'] == lab_).sum()), random_state=seed))
    s = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    print('抽样 %d 行 | %s' % (len(s), s['label'].value_counts().to_dict()))

    tr, ev = [], []
    for lab_, g in s.groupby('label'):
        idx = g.sample(frac=1, random_state=seed).index
        cut = int(len(idx) * 0.8)
        tr += list(idx[:cut])
        ev += list(idx[cut:])
    train = s.loc[sorted(tr), ['text', 'label']].reset_index(drop=True)
    evald = s.loc[sorted(ev), ['text', 'label']].reset_index(drop=True)
    train.to_csv(f'{OUT}/train.csv', index=False)
    evald.to_csv(f'{OUT}/eval.csv', index=False)

    # 泄漏/难度画像：正文里是否直接出现方向词
    ev_keys = ['增长', '增加', '上升', '盈利', '减少', '下降', '亏损', '扭亏']
    hit = evald['text'].apply(lambda x: any(k in x for k in ev_keys))
    print('\ntrain %d / eval %d | eval 里正文含方向词的占 %.1f%%（越高越"读数字"，越不像语义分类）' % (
        len(train), len(evald), hit.mean() * 100))
    print('文本长度 中位 %d / 95%% %d' % (train['text'].str.len().median(), train['text'].str.len().quantile(.95)))

    json.dump({'n': len(s), 'train': len(train), 'eval': len(evald),
               'dist': train['label'].value_counts().to_dict(),
               'eval_dist': evald['label'].value_counts().to_dict(),
               'direction_word_ratio': float(hit.mean())},
              open(f'{OUT}/meta.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('→', OUT)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    with RunLogger(__file__, tag='laya_prep'):
        main(a.n, a.seed)
