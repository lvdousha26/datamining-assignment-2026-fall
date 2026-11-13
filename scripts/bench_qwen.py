# -*- coding: utf-8 -*-
"""Qwen3-0.6B 两条路线：标签 logit（Jev-like）vs 生成式 SFT

路线 A（logit）：不算完整句子，只取「利好/利空/中性」三个候选串在答案位置的
               对数似然，softmax 后 3 分类。快、稳、可校准——Jev 的做法。
路线 B（SFT） ：正常自回归，只在答案段算 loss，推理时贪心生成再解析标签。

为什么不用 LLaMA-Factory：它要拉 datasets/accelerate/trl/transformers 一整套，
而这个环境历史上被 `pip install` 连带升级过 torch（CUDA 消失）。这里用
transformers + peft 手写等价流程（LoRA r=8，答案段 CE），超参与 LLaMA-Factory 对齐。

用法：
  python scripts/bench_qwen.py --mode logit
  python scripts/bench_qwen.py --mode sft
产出：runs/laya_yjyg/bench_qwen_<mode>.json + pred_qwen_<mode>.csv
"""
import argparse
import json
import os
import sys
import time

import os as _os
_os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
DIR = f'{ROOT}/runs/laya_yjyg'
LAB = ['利好', '利空', '中性']
L2I = {l: i for i, l in enumerate(LAB)}
MODEL = r'D:/quant/models/Qwen3-0.6B' if os.path.isdir(r'D:/quant/models/Qwen3-0.6B') else 'Qwen/Qwen3-0.6B'

SYS = '你是A股分析师。只回答一个词：利好、利空或中性。'
USER = ('下面是A股上市公司业绩预告的原文：\n{text}\n'
        '问题：这条业绩预告对股价的短期影响是利好、利空还是中性？')


def build_prompt(tok, text, with_answer=None):
    msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': USER.format(text=text)}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        s = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        s = tok.apply_chat_template(msgs, **kw)
    if with_answer is not None:
        s = s + with_answer
    return s


class LogitDS(Dataset):
    """把 prompt 与三个候选标签拼起来，返回 input_ids / 标签段的 mask"""

    def __init__(self, texts, labels, tok, max_len):
        self.items = []
        for t, y in zip(texts, labels):
            p = build_prompt(tok, t)
            lab_ids = [tok(l, add_special_tokens=False)['input_ids'] for l in LAB]
            ids = tok(p, add_special_tokens=False)['input_ids']
            self.items.append((ids, lab_ids, L2I[y]))
        self.tok, self.ml = tok, max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, lab_ids, y = self.items[i]
        ids = ids[: self.ml - 6]
        return {'ids': torch.tensor(ids), 'y': y}


def collate_logit(batch, tok, lab_ids):
    ml = max(len(b['ids']) for b in batch)
    B = len(batch)
    inp = torch.full((B, ml), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((B, ml), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b['ids'])
        inp[i, :n] = b['ids']
        att[i, :n] = 1
    return inp, att, torch.tensor([b['y'] for b in batch])


def label_scores(model, inp, att, lab_ids, max_lab=3):
    """对每个样本，算三个候选标签串在 prompt 末尾的对数似然均值

    做法：把 prompt 和标签拼接后一次前向，取标签区间对应位置的 log_softmax。
    为了批量，三种标签分开跑（3 次前向）。
    """
    B, L = inp.shape
    scores = inp.new_zeros((B, len(lab_ids)), dtype=torch.float32)
    for k, ids in enumerate(lab_ids):
        n = len(ids)
        full = torch.cat([inp, torch.tensor([ids], dtype=torch.long).repeat(B, 1).to(inp.device)], 1)
        m = torch.cat([att, torch.ones((B, n), dtype=torch.long, device=att.device)], 1)
        out = model(input_ids=full, attention_mask=m).logits
        logp = torch.log_softmax(out.float(), -1)
        # 标签第 j 个 token 的概率来自位置 L-1+j
        sc = []
        for j in range(n):
            tgt = torch.tensor(ids[j], device=inp.device)
            sc.append(logp[:, L - 1 + j, tgt])
        scores[:, k] = torch.stack(sc, 1).mean(1)
    return scores


class SFTDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.items, self.tok, self.ml = [], tok, max_len
        for t, y in zip(texts, labels):
            p = build_prompt(tok, t)
            pid = tok(p, add_special_tokens=False)['input_ids']
            lid = tok(y + tok.eos_token, add_special_tokens=False)['input_ids']
            ids = (pid + lid)[:max_len]
            lab = [-100] * min(len(pid), len(ids)) + lid
            lab = lab[:len(ids)]
            self.items.append((ids, lab))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, lab = self.items[i]
        return {'ids': torch.tensor(ids), 'lab': torch.tensor(lab)}


def collate_sft(batch, tok):
    ml = max(len(b['ids']) for b in batch)
    B = len(batch)
    inp = torch.full((B, ml), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((B, ml), dtype=torch.long)
    lab = torch.full((B, ml), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b['ids'])
        inp[i, :n] = b['ids']
        att[i, :n] = 1
        lab[i, :n] = b['lab']
    return inp, att, lab


def main(mode='logit', epochs=3, batch=2, accum=8, lr=1e-4, max_len=192):
    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev)
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                     'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False

    tr = pd.read_csv(f'{DIR}/train.csv')
    ev = pd.read_csv(f'{DIR}/eval.csv')
    lab_ids = [tok(l, add_special_tokens=False)['input_ids'] for l in LAB]
    print('%s | mode=%s | train %d / eval %d | 标签 token %s' % (
        MODEL, mode, len(tr), len(ev), [len(x) for x in lab_ids]), flush=True)

    if mode == 'logit':
        ds = LogitDS(tr.text, tr.label, tok, max_len)
        dl = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=lambda b: collate_logit(b, tok, lab_ids))
        dev_ds = LogitDS(ev.text, ev.label, tok, max_len)
        dev_dl = DataLoader(dev_ds, batch_size=8, collate_fn=lambda b: collate_logit(b, tok, lab_ids))
    else:
        ds = SFTDS(tr.text, tr.label, tok, max_len)
        dl = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=lambda b: collate_sft(b, tok))
        dev_ds = SFTDS(ev.text, ev.label, tok, max_len)
        dev_dl = DataLoader(dev_ds, batch_size=8, collate_fn=lambda b: collate_sft(b, tok))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total = len(dl) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    step, t0, best, best_pred = 0, time.time(), 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        for bi, b in enumerate(dl):
            inp, att = b[0].to(dev), b[1].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if mode == 'logit':
                    sc = label_scores(model, inp, att, lab_ids)
                    loss = torch.nn.functional.cross_entropy(sc, b[2].to(dev))
                else:
                    out = model(input_ids=inp, attention_mask=att, labels=b[2].to(dev))
                    loss = out.loss
            (loss / accum).backward()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print('  ep%d %d/%d loss %.4f | %.0fs | 显存 %.1fG' % (
                    ep, step, total, loss.item(), time.time() - t0,
                    torch.cuda.memory_allocated() / 1e9), flush=True)

        # ---- eval ----
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            if mode == 'logit':
                for b in dev_dl:
                    inp, att = b[0].to(dev), b[1].to(dev)
                    sc = label_scores(model, inp, att, lab_ids)
                    preds += [LAB[i] for i in sc.argmax(-1).tolist()]
                    ys += [LAB[i] for i in b[2].tolist()]
            else:
                # 逐条生成：右 padding 会让 decoder 从 pad 位置接着写，批量生成结果不可用
                for k, (x, lab) in enumerate(zip(ev.text, ev.label)):
                    pr = build_prompt(tok, x)
                    ids = tok(pr, add_special_tokens=False, return_tensors='pt').to(dev)
                    out = model.generate(input_ids=ids['input_ids'],
                                         attention_mask=ids['attention_mask'],
                                         max_new_tokens=8, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                    txt = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
                    hit = next((l for l in LAB if l in txt), '中性')
                    if k < 3:
                        print('    sample %d: %r -> %s (真值 %s)' % (k, txt[:30], hit, lab), flush=True)
                    preds.append(hit); ys.append(lab)
        acc = float(np.mean(np.array(preds) == np.array(ys)))
        print('  → epoch %d eval acc %.4f' % (ep, acc), flush=True)
        if acc > best:
            best, best_pred = acc, preds

    # 单条延迟
    times = []
    for x in ev.text.tolist()[:40]:
        t = time.time()
        p = build_prompt(tok, x)
        ids = tok(p, add_special_tokens=False, return_tensors='pt').to(dev)
        with torch.no_grad():
            if mode == 'logit':
                label_scores(model, ids['input_ids'], ids['attention_mask'], lab_ids)
            else:
                model.generate(input_ids=ids['input_ids'], attention_mask=ids['attention_mask'],
                               max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        times.append((time.time() - t) * 1000)
    lat = float(np.median(times))

    try:
        model.save_pretrained(f'{DIR}/qwen_{mode}')
    except Exception as e:
        print('保存 adapter 失败:', e)

    out = {'model': MODEL, 'mode': mode, 'acc': best, 'ms': lat, 'epochs': epochs,
           'train_s': time.time() - t0, 'lora': 'r8'}
    json.dump(out, open(f'{DIR}/bench_qwen_{mode}.json', 'w'), ensure_ascii=False, indent=1)
    pd.DataFrame({'text': ev.text, 'label': ev.label, 'pred': best_pred}).to_csv(
        f'{DIR}/pred_qwen_{mode}.csv', index=False)
    print('\n[QWEN %s] acc %.4f | %.1f ms/条 | 训练 %.0fs' % (mode, best, lat, out['train_s']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='logit', choices=['logit', 'sft'])
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    a = ap.parse_args()
    with RunLogger(__file__, tag='qwen_' + a.mode):
        main(a.mode, a.epochs, a.batch, a.accum, a.lr)
