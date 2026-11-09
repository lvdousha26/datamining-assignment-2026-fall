# -*- coding: utf-8 -*-
"""远程部署 v2：传 raw 行情 + 脚本，远程重算面板再训练

上传速度实测 0.5 MB/s → 不传 2.5GB 的 arr.npz，改传 271MB 原始行情自己算。
"""
import os, sys, time, glob
import paramiko

HOST = os.environ.get('RHOST', '')
PORT = int(os.environ.get('RPORT', '22'))
USER = os.environ.get('RUSER', 'root')
PW = os.environ['RHPW']
RD = '/root/autodl-tmp/quant'
LOCAL = 'D:/quant'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
          banner_timeout=40, auth_timeout=40, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()


def run(cmd, t=3600):
    i, o, e = c.exec_command(cmd, timeout=t)
    out = o.read().decode(); err = e.read().decode()
    return out, err


def up(local, remote):
    sz = os.path.getsize(local)
    t = time.time()
    sftp.put(local, remote)
    return sz, time.time() - t


os.makedirs if False else None
# 建目录
run(f'mkdir -p {RD}/scripts {RD}/data/big/raw {RD}/checkpoints {RD}/reports')

# 1) 脚本
scripts = ['big_features.py', 'big_train.py', 'tf_net.py', 'stack_ridge.py', 'muon_optimizer.py']
for s in scripts:
    up(f'{LOCAL}/scripts/{s}', f'{RD}/scripts/{s}')
print('脚本已传 %d 个' % len(scripts), flush=True)

# 2) 元数据
for f in ['st_calendar.parquet', 'ind_roll.parquet']:
    up(f'{LOCAL}/data/big/{f}', f'{RD}/data/big/{f}')
up(f'{LOCAL}/checkpoints/lgbm_meta.json', f'{RD}/checkpoints/lgbm_meta.json')
print('元数据已传', flush=True)

# 3) 行情分片
shards = sorted(glob.glob(f'{LOCAL}/data/big/raw/shard_*.parquet'))
tot = sum(os.path.getsize(f) for f in shards)
print('待传 %d 个 shard / %.0f MB' % (len(shards), tot / 1e6), flush=True)
t0 = time.time(); done = 0
for k, f in enumerate(shards, 1):
    sz, dt = up(f, f'{RD}/data/big/raw/{os.path.basename(f)}')
    done += sz
    el = time.time() - t0
    print('  [%d/%d] %s %.1fMB  %.1f/%.0fMB  %.1f分' %
          (k, len(shards), os.path.basename(f), sz / 1e6, done / 1e6, tot / 1e6, el / 60), flush=True)

print('\n传输完成 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)
o, e = run(f'du -sh {RD}/data/big/raw; ls {RD}/data/big/raw | wc -l')
print(o, e)
sftp.close(); c.close()
