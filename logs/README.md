# 日志规范

> 2026-11-09 建立。此前的运行日志散落在 `reports/*.log`，共 56 个，命名混乱
> （`big_train.log / big_train2.log / big_train3.log / big_train_v2.log / big_train_hfq.log / big_train_hfq_v2.log`），
> 且没有任何元信息——三个月后无法判断哪一版是最终结果。这里重新定规则。

---

## 一、谁写什么

| 类型 | 位置 | 内容 | 维护者 |
|---|---|---|---|
| **运行日志** | `logs/` | 脚本执行：参数、硬件、输出、耗时、状态 | `runlog.py` 自动 |
| **工作日志** | `.workbuddy/memory/YYYY-MM-DD.md` | 每天做了什么、结论、踩的坑 | Agent 手写 |
| **项目备忘** | `.workbuddy/memory/MEMORY.md` | 跨会话的长期约定 | Agent 手写 |
| **研究报告** | `reports/*.md` | 结论性文档（给人看） | 人工整理 |

**不再把运行日志写进 `reports/`。** 历史文件保留不动。

---

## 二、每次运行产出三个文件

```
logs/20261109_182527_build_data_dict.log    完整输出（header + 正文 + footer）
logs/20261109_182527_build_data_dict.json   结构化元数据（机器读）
logs/index.csv                              追加一行汇总（检索用）
```

`.log` 的样子：

```
==============================================================================
run_id    : 20261109_182527_build_data_dict
script    : build_data_dict.py
tag       : -
started   : 2026-11-09 18:25:27
argv      : -
cwd       : D:\quant
python    : 3.13.14 | Windows-11 | 16 cores
gpu       : NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
==============================================================================
扫描 panel_big…
...
==============================================================================
finished  : 2026-11-09 18:25:38
elapsed   : 11.5s
status    : OK
==============================================================================
```

`index.csv` 的列：`run_id, script, tag, start, elapsed_s, status, argv, host, log`

---

## 三、怎么接入（两行）

在脚本的 `ROOT = ...` 之前插入（已有 8 个脚本接好，见下）：

```python
if __name__ == '__main__':
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from runlog import RunLogger as _RL
        _RL(__file__).__enter__()
    except Exception:
        pass
```

或者用上下文管理器（更精确，异常也能捕获）：

```python
from runlog import RunLogger

if __name__ == '__main__':
    with RunLogger(__file__, tag='rebuild'):
        main()
```

**已接入**：`event_study` / `make_events` / `fetch_yjyg` / `build_data_dict` /
`big_features` / `big_train` / `evaluate_screen` / `purged_cv`

---

## 四、检索

```bash
# 最近 10 次运行
tail -10 logs/index.csv

# 所有失败
grep FAIL logs/index.csv

# 某脚本的全部运行历史
grep big_train logs/index.csv

# 某次运行的完整输出
cat logs/20261109_182527_build_data_dict.log

# 找出跑得最久的几次（按耗时排序）
python -c "
import pandas as pd
d = pd.read_csv('logs/index.csv')
print(d.nlargest(10,'elapsed_s')[['run_id','elapsed_s','status']].to_string(index=False))
"
```

---

## 五、规则

1. **不覆盖**：每次运行都是新文件。想比较两次结果，比 `run_id`。
2. **不手动删**：`status=FAIL` 的日志尤其要留，那是排错线索。
3. **tag 用于区分同脚本的不同用途**，比如 `--tag v2`、`--tag ablation`。
4. **长任务必留**：训练、重建面板、抓数据（>5 分钟）必须有日志。
5. **短任务可选**：一次性的检查脚本可以用 `with RunLogger(...)` 也可以不接。

---

## 六、清理策略

| 时间 | 动作 |
|---|---|
| 每 30 天 | 把 >30 天且 `status=OK` 的 `.log` 压缩归档到 `logs/archive/`（`.json` 保留） |
| 每 90 天 | `archive/` 里 >90 天的整体删除 |
| 永不删 | `index.csv`（只有几十 KB，是最完整的索引） |

---

## 七、历史日志（2026-11-09 之前）

`reports/*.log` 共 56 个，命名不规范、无元信息，**只作为历史参考，不再新增**。
其中较重要的几次（`train_hfq_v2.log`、`rebuild_hfq_v2.log`、`purged_cv_hfq2.log`）已在
`reports/data_fix_v2_report.md` 里被引用，不要删。
