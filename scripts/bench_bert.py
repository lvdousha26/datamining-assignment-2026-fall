# -*- coding: utf-8 -*-
"""BERT-base-chinese 微调（传统最优解基线）

不装 accelerate/Trainer，手写训练循环：依赖少、行为可见、显存可控。
用法：python scripts/bench_bert.py [--epochs 4] [--batch 32]
产出：runs/laya_yjyg/bench_bert.json + runs/laya_yjyg/bert/
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
DIR = f'{ROOT}/runs/laya_yjyg'
LAB = ['利好', '中性', '利空']
L2I = {l: i for i, l in enumerate(LAB)}
NAME = 'bert-base-chinese'


class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.t = list(texts)
        self.y = [L2I[x] for x in labels]
        self.tok = tok
        self.ml = max_len

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, max_length=self.ml, padding='max_length',
                     return_tensors='pt')
        return {'input_ids': e['input_ids'][0], 'attention_mask': e['attention_mask'][0],
                'labels': torch.tensor(self.y[i])}


def run_eval(model, loader, dev):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for b in loader:
            ids = b['input_ids'].to(dev)
            am = b['attention_mask'].to(dev)
            out = model(input_ids=ids, attention_mask=am).logits
            preds += out.argmax(-1).cpu().tolist()
            ys += b['labels'].tolist()
    return float(np.mean(np.array(preds) == np.array(ys))), preds, ys


def main(epochs=4, batch=32, lr=3e-5, max_len=128, data_dir=None, tag='bert'):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dd = data_dir or DIR
    tr = pd.read_csv(f'{dd}/train.csv')
    ev = pd.read_csv(f'{dd}/eval.csv')
    ycol = 'label' if 'label' in tr.columns else 'lab3'
    tr = tr.rename(columns={ycol: 'label'})
    ev = ev.rename(columns={ycol: 'label'})
    os.makedirs(f'{DIR}/{tag}', exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=3).to(dev)
    print('%s | 参数量 %.1fM | train %d / eval %d' % (
        NAME, sum(p.numel() for p in model.parameters()) / 1e6, len(tr), len(ev)), flush=True)

    dtr = DataLoader(DS(tr.text, tr.label, tok, max_len), batch_size=batch, shuffle=True)
    dev_loader = DataLoader(DS(ev.text, ev.label, tok, max_len), batch_size=64)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(dtr) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))

    best, best_pred = 0.0, None
    step, t0 = 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for b in dtr:
            ids, am = b['input_ids'].to(dev), b['attention_mask'].to(dev)
            y = b['labels'].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                loss = model(input_ids=ids, attention_mask=am, labels=y).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 20 == 0:
                print('  ep%d %d/%d loss %.4f | %.0fs | 显存 %.1fG' % (
                    ep, step, total, loss.item(), time.time() - t0,
                    torch.cuda.memory_allocated() / 1e9 if dev == 'cuda' else 0), flush=True)
        acc, pred, ys = run_eval(model, dev_loader, dev)
        print('  → epoch %d eval acc %.4f' % (ep, acc), flush=True)
        if acc > best:
            best, best_pred = acc, pred
            model.save_pretrained(f'{DIR}/{tag}')
            tok.save_pretrained(f'{DIR}/{tag}')

    # 单条延迟（batch=1，含 tokenize）
    model.eval()
    times = []
    for x in ev.text.tolist()[:100]:
        t = time.time()
        e = tok(x, truncation=True, max_length=max_len, return_tensors='pt').to(dev)
        with torch.no_grad():
            model(**e)
        times.append((time.time() - t) * 1000)
    lat = float(np.median(times))

    out = {'model': NAME, 'acc': best, 'ms': lat, 'epochs': epochs, 'batch': batch,
           'train_s': time.time() - t0, 'params_m': sum(p.numel() for p in model.parameters()) / 1e6}
    out['data_dir'] = dd
    json.dump(out, open(f'{DIR}/bench_{tag}.json', 'w'), ensure_ascii=False, indent=1)
    pd.DataFrame({'text': ev.text, 'label': ev.label, 'pred': [LAB[i] for i in best_pred]}).to_csv(
        f'{DIR}/pred_{tag}.csv', index=False)
    print('\n[BERT] acc %.4f | %.1f ms/条 | 训练 %.0fs' % (best, lat, out['train_s']))
    print('→', f'{DIR}/bench_bert.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--tag', default='bert')
    a = ap.parse_args()
    with RunLogger(__file__, tag=a.tag):
        main(a.epochs, a.batch, a.lr, data_dir=a.data_dir, tag=a.tag)
