# -*- coding: utf-8 -*-
"""在独立 venv 里跑 ReCAP，并把输出纳入项目 logs/ 规范

用法：
  python scripts/recap_run.py --dataset DOW30
  python scripts/recap_run.py --dataset DOW30 --pretrain-timesteps 2048 \
      --finetune-timesteps 256 --n-steps 256 --batch-size 64 --gate-epochs 5

说明：外部仓库在 D:/quant/recap/（已 gitignore），venv 在 D:/quant/recap_venv/。
子进程输出逐行回显，RunLogger 才能同时写进 logs/*.log。
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import RunLogger  # noqa: E402

ROOT = r'D:/quant'
VENV_PY = f'{ROOT}/recap_venv/Scripts/python.exe'
RECAP_DIR = f'{ROOT}/recap'


def main():
    args = sys.argv[1:] or ['--dataset', 'DOW30']
    cmd = [VENV_PY, 'main.py'] + args
    print('命令: %s' % ' '.join(cmd), flush=True)
    p = subprocess.Popen(
        cmd, cwd=RECAP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end='', flush=True)
    rc = p.wait()
    print('\n返回码: %d' % rc, flush=True)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == '__main__':
    with RunLogger(__file__, tag='recap'):
        main()
