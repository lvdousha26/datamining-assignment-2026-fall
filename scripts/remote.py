# -*- coding: utf-8 -*-
"""极简远程执行 / SFTP 上传（paramiko，密码由环境变量 RHPW 提供）"""
import os, sys, time, stat
import paramiko

HOST = os.environ.get('RHOST', '')
PORT = 37416
USER = 'root'
PW = os.environ.get('RHPW')
if not PW:
    sys.exit('need RHPW env')

_c = None


def conn():
    global _c
    if _c is None:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
                  banner_timeout=30, auth_timeout=30)
        _c = c
    return _c


def run(cmd, timeout=600, quiet=False):
    c = conn()
    ch = c.get_transport().open_session()
    ch.settimeout(timeout)
    ch.exec_command(cmd)
    out = b''
    while True:
        if ch.recv_ready():
            out += ch.recv(65536)
        elif ch.exit_status_ready():
            break
        else:
            time.sleep(0.05)
    while ch.recv_ready():
        out += ch.recv(65536)
    code = ch.recv_exit_status()
    txt = out.decode('utf-8', 'replace')
    if not quiet:
        print(txt, end='')
    return code, txt


def put(local, remote, verbose=True):
    c = conn()
    sftp = c.open_sftp()
    total = os.path.getsize(local)
    t0 = time.time()
    last = [0]

    def cb(done, tot):
        pct = done / max(tot, 1) * 100
        if pct - last[0] >= 5 or done == tot:
            sp = done / max(time.time() - t0, 1e-6) / 1e6
            print('  %5.1f%%  %.2f/%.2f GB  %.1f MB/s' % (pct, done / 1e9, tot / 1e9, sp), flush=True)
            last[0] = pct

    sftp.put(local, remote, callback=cb)
    sftp.close()
    if verbose:
        print('  done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        code, _ = run(' '.join(sys.argv[2:]))
        sys.exit(code)
    elif sys.argv[1] == 'put':
        put(sys.argv[2], sys.argv[3])
