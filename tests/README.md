# 测试

> 2026-11-09 建立｜35 个用例｜全量 10.2 秒

## 为什么这么写

**每个测试对应一个真实踩过的 bug**，不追求覆盖率。

这个项目 83 个脚本、8 条被证伪的路径，一路踩的坑比写的代码还多。如果测试只是
「跑一遍看有没有报错」，那等于没测。所以这里的原则是：

> 测试失败 = 你刚刚又踩了以前踩过的坑。

比如 `test_inf_does_not_spread` 对应那个 23 万条 inf 污染（停牌股 close=0 → 未来收益分母 0
→ 截面去均值把 ±inf 扩散给全市场）；`test_ridge_matches_ols_when_lambda_small` 对应
「lam 一大系数掉 15 倍」那次误判（实际是 rhs 用了未标准化的 y）。

## 跑

```bash
# 默认：只跑快的（30 个，3.6 秒）
python -m pytest

# 加上全量数据契约（35 个，10.2 秒）
python -m pytest -m ""

# 只跑慢的
python -m pytest -m slow

# 只跑某个主题
python -m pytest -k ridge
python -m pytest -k break

# 详细输出
python -m pytest -v
```

## 四个文件

| 文件 | 内容 | 速度 |
|---|---|---|
| `test_logic.py` | 纯逻辑，合成数据，无 IO | 3.4 s |
| `test_data_contract.py` | 真实面板的契约（形状/有限性/裁剪/标签） | 7 s |
| `test_screen_turnover.py` | 清单更替率口径（4 个，含变异测试） | 0.04 s |
| `conftest.py` | fixture：合成行情、真实面板（session 级只加载一次） | — |

## 覆盖了什么

**逻辑层（16 个）**

| 测试 | 对应的坑 |
|---|---|
| `test_stop_price_not_minus_100` | 停牌 close=0 → fwd5=−100% 假收益（踩过两次） |
| `test_inf_does_not_spread` | 单个 inf 扩散到全市场 23 万条 |
| `test_demean_sums_to_zero` | 截面去均值 |
| `test_rank_transform_bounded` | rank 标签必须落在 (0,1) |
| `test_limit_rule` | 涨跌停制度（主板/创业板分界 2020-08-24/科创/北交） |
| `test_cross_break_label_void` | 跨断层标签作废（假动量 0.7σ） |
| `test_car_geometric` | CAR 必须是几何超额，不能简单相减 |
| `test_car_out_of_range` | 窗口越界返回 nan |
| `test_placebo_monotonic` | 安慰剂分位单调（旧脚本把 95.6% 算成 52%） |
| `test_ridge_matches_ols_when_lambda_small` | rhs 用错 y（系数缩水 15 倍的假象） |
| `test_ridge_shrinks_with_lambda` | lam 增大系数范数单调不增 |
| `test_weight_align_by_index` | 权重按位置相减 → 错位（换手算成 100%） |
| `test_runlog_creates_three_files` | 日志三件套 |
| `test_runlog_records_failure` | 异常必须记 FAIL |

**契约层（15 个）**

- 面板形状与 feats 一致、因子名不重复、日期严格递增、代码 6 位补零
- 训练段早于 lock_start、封存段标签为空（物理隔离）
- **`test_lockbox_state_not_rerun`**：`runs <= 1`，防止未来的自己手滑重跑
- `daily_ret` 的 symbols 与面板**逐位一致**（差 6 只就整体错位）
- 日收益 `|r| ≤ 50%`、无 inf
- 数据字典无空白释义、事件表 schema、清单 D1 占比≈10%
- 关键报告不丢
- 慢：掩码内全因子有限、`|X| ≤ 4`、Y2 截面均值、Y2 标准差 ∈ (0.04, 0.10)

**口径层（4 个，`test_screen_turnover.py`）**

| 测试 | 对应的坑 |
|---|---|
| `test_churn_is_name_based_not_row_based` | 用 DataFrame 行索引算 D1 更替 → 跨日不重叠 → 每期恒 100% |
| `test_identical_list_has_zero_churn` | 名单完全没变也该是 0%，行索引版本会报 100% |
| `test_missing_symbol_index_raises` | 少了 `i` 列必须报错，不能悄悄退回行索引 |
| `test_evaluate_screen_goes_through_d1_churn` | 防止有人把 `set(...index)` 写回脚本 |

这个坑的真实代价：`screen_eval.json` 里换手 97.9%（实际 44%），成本每期多扣 1.7bp，
年化超额被低估 0.8pp（2024 +4.99% → +5.85%，2025H1 +3.83% → +4.67%）。
用例做过变异验证：把 `['i']` 改回 `.index`，前两条立刻失败（1.0 vs 0.0）。

## 已知特性（不是 bug）

**`Y2` 的截面均值不严格为 0**（中位 1.0e-3、max 2.1e-2）。
因为去均值是在 `ok` 过滤（成交额 / 上市满 250 日 / 非 ST）**之前**做的，
过滤后子集必然残留偏差。对 rank 训练无实质影响，所以不改管线；
但测试会盯着它——中位数一旦涨到 5e-3 以上，说明过滤逻辑变了。

## CI（GitHub Actions）

`.github/workflows/ci.yml` 在每次 push / PR 自动跑：

- Python 3.12 与 3.13 双版本矩阵，只装 `requirements-ci.txt`（numpy/pandas/scipy/pytest，不含 torch）
- CI 上的结果：**23 passed / 7 skipped / 5 deselected**。7 个 skip 是依赖 `data/big/*.npz` 的契约测试
  （仓库不跟踪 2.5 GB 面板，fixture 自动 skip），5 个 deselected 是 `-m "not slow"`
- 打 `v*` tag 会额外触发 Release 任务：把 README 与 `reports/` 打包成 zip 挂到 Release

路径不写死：`conftest.py` / `test_data_contract.py` / `test_screen_turnover.py` 的项目根目录
默认取仓库位置，也可用环境变量 `QUANT_ROOT` 覆盖，Windows 本地与 Linux runner 通用。

本地跑（数据齐时能跑满 30 个）：

```bash
python -m pytest              # 快速集
python -m pytest -m ""        # 含慢契约，35 个
```

## 加测试的约定

修完一个 bug，就在 `test_logic.py` 里加一条，docstring 写三件事：

1. **现象**（什么错）
2. **历史**（什么时候踩的、后果多大）
3. **期望**（修好之后应该是什么样）

```python
def test_xxx():
    """一句话说清这个坑

    2026-11-xx 踩过：具体现象 + 影响范围
    """
```

这样半年后再看，测试文件本身就是一份 bug 档案。
