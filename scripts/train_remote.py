# -*- coding: utf-8 -*-
"""远程执行：改路径 → 跑 big_features → 跑 big_train（远程 nohup 后台，日志轮询）

用法：
  python train_remote.py prep      # sed 改 ROOT + 环境自检
  python train_remote.py feat      # 启动 big_features
  python train_remote.py train     # 启动 big_train
  python train_remote.py log [n]   # 看日志尾部
  python train_remote.py stat      # 看产物与显存
"""
import os, sys
import paramiko

HOST = os.environ.get('RHOST', '')
PORT = int(os.environ.get('RPORT', '22'))
USER = os.environ.get('RUSER', 'root')
PW = os.environ['RHPW']
RD = '/root/autodl-tmp/quant'
PY = '/root/miniconda3/bin/python'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
          banner_timeout=40, auth_timeout=40, allow_agent=False, look_for_keys=False)


def run(cmd, t=300):
    i, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(), e.read().decode()


act = sys.argv[1] if len(sys.argv) > 1 else 'stat'

if act == 'prep':
    o, e = run(f"sed -i 's|D:/quant|{RD}|g' {RD}/scripts/*.py && grep -n \"^ROOT\" {RD}/scripts/big_features.py {RD}/scripts/big_train.py")
    print(o, e)
    o, e = run(f'cd {RD} && ls data/big/raw | wc -l && du -sh data/big/raw && {PY} -c "import pandas as pd;d=pd.read_parquet(\'data/big/raw/shard_000.parquet\');print(d.shape,d.columns.tolist())"')
    print(o, e)

elif act in ('feat', 'train'):
    script = 'big_features.py' if act == 'feat' else 'big_train.py'
    log = 'feat.log' if act == 'feat' else 'train.log'
    o, e = run(f'cd {RD} && nohup {PY} scripts/{script} > reports/{log} 2>&1 & echo PID=$!')
    print(script, 'started:', o.strip(), e.strip()[:200])

elif act == 'log':
    n = sys.argv[2] if len(sys.argv) > 2 else '25'
    o, e = run(f'cd {RD} && ls -la reports/*.log 2>/dev/null | tail -3; echo "--- feat ---"; tail -{n} reports/feat.log 2>/dev/null; echo "--- train ---"; tail -{n} reports/train.log 2>/dev/null')
    print(o, e)

elif act == 'stat':
    o, e = run(f'cd {RD} && ps aux | grep -c "[b]ig_" ; ls -la data/big/arr.npz data/big/lockbox_labels.npz checkpoints/ 2>/dev/null; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader')
    print(o, e)

elif act == 'kill':
    o, e = run('pkill -f big_features.py; pkill -f big_train.py; echo killed')
    print(o, e)

c.close()
