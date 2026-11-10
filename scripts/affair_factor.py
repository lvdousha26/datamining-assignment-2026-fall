# -*- coding: utf-8 -*-
"""CEO 丑闻反弹检验 —— 通用事件研究框架的第一个实例

历史
----
本文件是项目里最早手写的事件研究（AMZN 贝索斯、MSFT 盖茨），也是「安慰剂分位」这个方法的起点。
但它的安慰剂实现有 bug：MSFT 报 52%，正确值 95.6%。
2026-11-09 起，计算逻辑全部迁到 `scripts/event_study.py`；本文件只保留**事件定义与同侪名单**，
其余交给框架执行——一套引擎、多处调用，避免两份实现各自漂移。

待检验命题（网络梗）
------------------
    AMZN 自 2019 贝索斯婚外情曝光后市值 +2 万亿美元
    MSFT 自 2021 盖茨出轨曝光后市值 +2 万亿美元
    → 「CEO 出丑 = 利好」

框架会自动给出三件对照，这正是原脚本缺的两件：
    ① 多窗口 CAR（几何超额，减纳指）
    ② 安慰剂分位（该股全部历史同长度窗口，全样本不抽样）
    ③ 同侪排序（同日起算的其他大盘股）

用法
----
    python scripts/affair_factor.py            # 跑这两个事件，输出 reports/ev_affair.*
    python scripts/affair_factor.py --dry      # 只写事件表，不执行
"""
import argparse
import os
import subprocess
import sys

ROOT = r'D:/quant'
EVENTS = [('AMZN', '2019-01-09', 'CEO丑闻', '贝索斯婚外情曝光（National Enquirer）'),
          ('MSFT', '2021-05-16', 'CEO丑闻', '盖茨离婚/董事会调查曝光（WSJ）')]
PEERS = ['AAPL', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'WMT', 'XOM', 'V']


def main(dry=False):
    os.makedirs(f'{ROOT}/events', exist_ok=True)
    ev_csv = f'{ROOT}/events/affair.csv'
    with open(ev_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('symbol,date,label,note\n')
        for s, d, lab, note in EVENTS:
            f.write('%s,%s,%s,%s\n' % (s, d, lab, note))
    peer_txt = f'{ROOT}/events/affair_peers.txt'
    open(peer_txt, 'w', encoding='utf-8', newline='').write('\n'.join(PEERS) + '\n')
    print('事件表 → %s（%d 个事件）\n同侪 → %s（%d 只）' % (ev_csv, len(EVENTS), peer_txt, len(PEERS)))

    cmd = [sys.executable, f'{ROOT}/scripts/event_study.py',
           '--events', ev_csv, '--market', 'us', '--peers', peer_txt,
           '--windows', '1,5,20,60,250,500', '--out', f'{ROOT}/reports/ev_affair']
    print('\n调用通用框架：\n  ' + ' '.join(cmd) + '\n')
    if dry:
        return
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print('框架执行失败（returncode=%d）' % r.returncode)
        sys.exit(r.returncode)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true', help='只生成事件表，不执行框架')
    a = ap.parse_args()
    main(a.dry)
