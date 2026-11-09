# -*- coding: utf-8 -*-
"""运行日志：给任何脚本加两行，就把「谁、什么时候、用什么参数、跑了多久、成没成功」记下来

用法（推荐，上下文管理器）
    from runlog import RunLogger

    if __name__ == '__main__':
        with RunLogger(__file__, tag='rebuild'):
            main()

用法（装饰器）
    from runlog import logged

    @logged
    def main():
        ...

产出（每次运行三个文件）
    logs/20261109_183012_event_study.log    完整输出（含 header/footer）
    logs/20261109_183012_event_study.json   结构化元数据
    logs/index.csv                          追加一行汇总，用于检索

约定
    - 日志目录：D:/quant/logs
    - 文件名：<YYYYMMDD_HHMMSS>_<脚本名>[_<tag>].log
    - 不覆盖：每次运行都是新文件（这正是旧日志 big_train2/3/_v2 混乱的根源）
    - stdout/stderr 原样保留（echo=True），不改变脚本行为
"""
import atexit
import csv
import json
import os
import platform
import socket
import sys
import time
import traceback

ROOT = r'D:/quant'
LOGDIR = f'{ROOT}/logs'
INDEX_COLS = ['run_id', 'script', 'tag', 'start', 'elapsed_s', 'status', 'argv', 'host', 'log']


def _gpu():
    try:
        import subprocess
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5)
        return (out.stdout or '').strip().split('\n')[0] or ''
    except Exception:
        return ''


def _host():
    return {
        'host': socket.gethostname(),
        'platform': f'{platform.system()}-{platform.release()}',
        'cpu_count': os.cpu_count(),
        'python': platform.python_version(),
        'gpu': _gpu(),
    }


class _Tee:
    """把 stdout 同时写到控制台和文件"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


class RunLogger:
    def __init__(self, script=None, tag='', echo=True, logdir=LOGDIR):
        self.script = os.path.basename(script) if script else (
            os.path.basename(sys.argv[0]) if sys.argv else 'interactive')
        self.tag = tag
        self.echo = echo
        self.logdir = logdir
        self.t0 = None
        self.run_id = None
        self.path = None
        self._fh = None
        self._old_out = None
        self._old_err = None

    # ---------- 上下文协议 ----------
    def __enter__(self):
        os.makedirs(self.logdir, exist_ok=True)
        self.t0 = time.time()
        ts = time.strftime('%Y%m%d_%H%M%S', time.localtime(self.t0))
        stem = f'{ts}_{os.path.splitext(self.script)[0]}'
        if self.tag:
            stem += f'_{self.tag}'
        self.run_id = stem
        self.path = f'{self.logdir}/{stem}.log'

        self._fh = open(self.path, 'w', encoding='utf-8', buffering=1)
        h = _host()
        head = [
            '=' * 78,
            f'run_id    : {self.run_id}',
            f'script    : {self.script}',
            f'tag       : {self.tag or "-"}',
            f'started   : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t0))}',
            f'argv      : {" ".join(sys.argv[1:]) or "-"}',
            f'cwd       : {os.getcwd()}',
            f'python    : {h["python"]} | {h["platform"]} | {h["cpu_count"]} cores',
            f'gpu       : {h["gpu"] or "-"}',
            '=' * 78,
        ]
        self._fh.write('\n'.join(head) + '\n')
        if self.echo:
            print('\n'.join(head), flush=True)

        # 文件始终记录完整输出；echo 只控制 header/footer 是否回显到控制台
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._old_out, self._fh)
        sys.stderr = _Tee(self._old_err, self._fh)

        # 顶层脚本（无 main / 无 with）崩溃时也能记成 FAIL
        self._excepthook = sys.excepthook

        def _hook(et, ev, tb):
            try:
                self._fh.write('\n[UNCAUGHT]\n' + ''.join(
                    traceback.format_exception(et, ev, tb)))
            except Exception:
                pass
            self._close(status=f'FAIL({getattr(et, "__name__", "Exception")})')
            self._excepthook(et, ev, tb)

        sys.excepthook = _hook
        atexit.register(self._close)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            msg = ''.join(traceback.format_exception(exc_type, exc, tb))
            try:
                self._fh.write('\n[EXCEPTION]\n' + msg)
            except Exception:
                pass
            status = f'FAIL({exc_type.__name__})'
        else:
            status = 'OK'
        self._close(status=status)
        return False        # 不吞异常

    # ---------- 收尾 ----------
    def _close(self, status=None):
        if self._fh is None:
            return
        if status is None:
            status = 'FAIL' if sys.exc_info()[0] else 'OK'
        el = time.time() - self.t0
        tail = [
            '=' * 78,
            f'finished  : {time.strftime("%Y-%m-%d %H:%M:%S")}',
            f'elapsed   : {el:.1f}s',
            f'status    : {status}',
            '=' * 78,
        ]
        try:
            sys.stdout, sys.stderr = self._old_out, self._old_err
            self._fh.write('\n'.join(tail) + '\n')
            self._fh.close()
            if self.echo:
                print('\n'.join(tail), flush=True)
        except Exception:
            pass
        finally:
            self._fh = None

        meta = dict(run_id=self.run_id, script=self.script, tag=self.tag,
                    start=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.t0)),
                    elapsed_s=round(el, 1), status=status,
                    argv=sys.argv[1:], log=os.path.basename(self.path or ''),
                    **_host())
        try:
            with open(self.path.replace('.log', '.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            idx = os.path.join(self.logdir, 'index.csv')
            new = not os.path.exists(idx)
            with open(idx, 'a', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=INDEX_COLS)
                if new:
                    w.writeheader()
                w.writerow({k: meta.get(k, '') for k in INDEX_COLS})
        except Exception:
            pass


def logged(fn):
    """装饰器版：@logged def main(): ..."""
    def wrap(*a, **kw):
        with RunLogger(sys.argv[0], tag=fn.__name__):
            return fn(*a, **kw)
    wrap.__name__ = fn.__name__
    return wrap


if __name__ == '__main__':
    with RunLogger(__file__, tag='selftest'):
        print('自检：这行应该同时出现在控制台和 logs/*.log 里')
        print(f'host = {_host()}')
    print(f'日志已写入 {LOGDIR}')
