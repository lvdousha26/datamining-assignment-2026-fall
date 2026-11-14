# ReCAP（KDD 2026）复现：跑通 DOW30 与环境兼容记录

> 2026-11-14｜论文：Chaofan Pan 等《Regime-Adaptive Continual Learning for Portfolio Management》，
> KDD 2026，arXiv:2606.00143，DOI 10.1145/3770855.3817620（西南财经大学等）
> 官方仓库：`https://github.com/Dumail/ReCAP`，README 自称「最小可运行实现」

## 一、结论先行

1. **跑通了**：独立 venv → DOW30 数据 → ARD 切出 **50 个训练段 / 20 个测试段** →
   PPO 预训练 → policy library 满 20 个策略向量 → regime-gate → CR / SR / MDD 全链路返回 0。
   smoke test **7/7 通过**。
2. **没有复现出论文的收益**：本次最好一组 CR **−55.1%**，论文报告的是正收益
   （论文 NAS100 匹配预算对照下 CR 164.89% / SR 1.14）。原因不是「方法无效」，是复现条件差了 40 倍（第四节）。
3. **唯一拿到的正面证据**：相同 5 万步预算下，**开启 `online_update`（论文核心的持续更新机制）
   把 CR 从 −99.28% 拉到 −55.11%（+44pp）、SR 从 −6.25 到 −0.54、最大回撤从 99.3% 到 70.4%**。
   机制本身在小预算下也确实在起作用。

## 二、环境（与主项目完全隔离）

| 项 | 值 |
|---|---|
| 仓库 / venv | `D:/quant/recap/`（已 gitignore）、`D:/quant/recap_venv/`（已 gitignore） |
| Python | 3.13.14 |
| 关键依赖 | torch 2.15.0 / stable-baselines3 2.9.2 / gymnasium 1.4.0 / finrl 0.3.7 / pandas 3.1.1 / numpy 2.5.4 / yfinance 1.8.0 |
| 数据 | yfinance，论文区间 2008-05-01 ~ 2025-04-29；**挂代理后可访问**（AAPL 试拉 9 行 / 5.7s） |
| 日志 | `scripts/recap_run.py` 逐行回显子进程输出，自动进 `logs/`（runlog 规范） |

**两个安装坑**：① 全局 pip 指向清华镜像，报 `No matching distribution found for numpy`，
换 `-i https://mirrors.aliyun.com/pypi/simple/` 后正常；② `requirements.txt` 漏了 finrl 的
隐式依赖（matplotlib / stockstats / scipy），且 finrl 0.3.7 的 `__init__` 会连锁导入
`alpaca_trade_api` 等交易 API——补包是无底洞，改用静态成分表绕过（见下）。

## 三、四处兼容补丁（只改外部仓库，不动算法）

| # | 文件 | 问题 | 修法 |
|---|---|---|---|
| 1 | `recap/data.py` `_calculate_turbulence` | pandas 3.1 的 `to_numpy()` 返回**只读视图**，源码却 `cov += np.eye(...)*1e-6` → `ValueError: output array is read-only`，6 个 smoke 全挂 | `to_numpy(dtype=np.float64, copy=True)` |
| 2 | `recap/recap.py` `_build_segment_tensors` | torch 2.15 对只读数组报 `not writable` 警告 | `np.array(..., dtype=np.float32, copy=True)` 后再 `as_tensor` |
| 3 | `recap/tickers.py` | `from finrl.config_tickers import ...` 触发 finrl 包初始化 → 缺 `alpaca_trade_api` | try/except 回退到 `recap/_tickers_data.py`（复制自 finrl 的纯静态常量文件，1432 行），**只取 3 个成分表常量，不改任何算法** |
| 4 | `recap/data.py` `_download_vix_frame` | yfinance 1.8 单标的也返回 **MultiIndex (Price, Ticker)** 两层列，`reset_index()[["Date","Close"]]` 导致 merge 报 `Not allowed to merge between different levels` | 用 `raw.xs("Close", axis=1, level="Price")` 拍平 |

补丁后 `pytest tests/test_smoke.py -q` → **7 passed**。
另外 `WBA`（Walgreens）已退市，yfinance 返回 404；`_keep_complete_trading_dates` 按实际可得标的对齐，
不影响跑通，但成分表是 **2021/10 静态快照**，与论文数据不完全一致。

## 四、DOW30 三次运行（全部返回码 0）

| # | 训练预算 | online_update | CR | SR | MDD | 耗时 |
|---|---|---|---|---|---|---|
| 1 冒烟 | 1 万步（pretrain 2048 / finetune 256、gate 5 epoch） | 关 | −100.00% | −37.63 | 100.00% | 165 s |
| 2 中等 | 5 万步（pretrain 约 3.7 万 / finetune 256、gate 20 epoch） | 关 | −99.28% | −6.25 | 99.32% | 190 s |
| 3 中等+持续更新 | 同上 | **开** | **−55.11%** | **−0.54** | 70.43% | 409 s |

日志：`logs/20261114_120707_recap_run_recap.log` / `_121018_` / `_121516_`。

**为什么净值几乎归零（不是 NaN bug）**：`env.step()` 每步按
`turnover = |w_t − w_{t−1}|.sum()` 全额收 `0.1% × turnover` 成本；
策略未收敛时 softmax 权重每日抖动，L1 换手接近 2 → 每步约 0.2% 成本，
测试期上千个交易日复利下来足以把净值磨到接近 0（代码里有 `max(portfolio_growth, 1e-6)` 兜底，
所以是稳定的指数衰减而非数值爆炸）。开启持续更新后权重抖动下降，亏损明显收窄——
这也反过来印证成本是主因。

## 五、为什么这些数字不能和论文比

1. **训练预算差 40 倍**：本次 5 万步，仓库默认 **200 万步**；论文在匹配预算的
   NAS100 对照里每阶段用 `10^4` 更新步。官方 README 明确说本仓库训练步数
   「显著少于论文完整实验」，目的是让回归验证在有限时间内跑完，不是复现结果；
2. **成本口径偏狠**：环境按每步 L1 权重变化全额计费，未收敛策略被过度惩罚；
3. **成分表静态**：2021/10 的 DOW30，WBA 已退市，缺一只标的；
4. **超参未对齐论文**：论文 κ=0.5σ、h=2.5σ、δs=0.5、PPO 全量预算、
   critic 每段重初始化；本次 gate_epochs=20、finetune=256/段，都没调。

CPU 上按 5 万步 190 s 线性外推，默认 200 万步约 **2 小时以上**；
本次以「跑通 + 趟平环境坑」为目标，没有跑满预算。

## 六、续跑命令

```bash
# 完整默认预算（2M 步，CPU 约 2h+）
python scripts/recap_run.py --dataset DOW30

# 开启持续更新（论文核心机制）
python scripts/recap_run.py --dataset DOW30 --online-update

# 快速回归
python scripts/recap_run.py --dataset DOW30 \
  --pretrain-timesteps 2048 --finetune-timesteps 256 \
  --n-steps 256 --batch-size 64 --gate-epochs 5 --total-training-steps 10000
```

所有运行自动写 `logs/`（含参数、耗时、成功状态）。`D:/quant/recap/` 与
`D:/quant/recap_venv/` 均已加入 `.gitignore`，主仓库不含第三方代码与虚拟环境。

## 七、复现判断

**跑通 ≠ 复现结果**。这份仓库提供了主流程与核心机制（ARD / policy library / RGM），
但自称最小实现，加上依赖版本大幅前进（pandas 3 / numpy 2 / torch 2.15 / yfinance 1.8），
需要 4 处兼容补丁才能启动。要得到论文量级的数字，得跑满 200 万步并把超参对齐，
再加上论文用的完整数据集（含 VIX/turbulence 的历史快照）。
本次价值在于：**环境、依赖、数据、日志链路全部打通并留档，下次可以直接跑正式预算**。
