# 数据挖掘技术与应用 · 期末报告代码仓库

**题目**：基于全市场 A 股面板的多因子选股模型——从 IC 检验到负向筛选的实证研究

**核心结论**：模型能稳定预测 5 日收益排序（5 折 CV IC **0.1041 ± 0.0094**，五折全正），
但预测力高度集中在「垫底那一撮」——A 股无法做空全市场，所以它只能做**减法**（禁买清单），
年化贡献 **+3~5%**。另外 8 条路径（SVM、多空、行业动量、基本面加成、Muon、TAA、策略轮动、业绩预告事件）
全部被证伪。

---

## 一、环境

```bash
pip install -r requirements.txt
python -m pytest              # 26 个快速测试，3.4 秒
python -m pytest -m ""        # 31 个（含数据契约），10 秒
```

- Python 3.13 / torch 2.7.1+cu118 / CUDA 可用
- 训练用 RTX 4060 8G（TF 训练约 25 分钟），大规模实验可换 4090/A800

## 二、目录

```
scripts/        85 个脚本（数据管线 / 训练 / 验证 / 事件研究 / 策略）
tests/          31 个测试（每个对应一个真实踩过的 bug）
reports/        46 份结论报告
events/         事件研究的事件表
live/           当期禁买清单
logs/           运行日志规范（日志本身不入库）
data/           数据（不入库，见第五节）
checkpoints/    模型权重（不入库）
```

## 三、快速开始

```bash
# 1. 重建面板（需要先有原始行情，见第五节）
python scripts/big_features.py          # 约 40 分钟，产出 arr.npz

# 2. 训练（LGBM + Transformer 3 seed + Ridge 元模型）
python scripts/big_train.py             # 约 25 分钟

# 3. 5 折 Purged CV
python scripts/purged_cv.py             # 约 10 分钟

# 4. 生成当期禁买清单
python scripts/live_screen.py

# 5. 事件研究（示例：业绩预告）
python scripts/make_events.py --type yjyg --per-year 15
python scripts/event_study.py --events events/yjyg.csv --market cn --lag 1

# 6. 关联规则
python scripts/assoc_rules.py --sample 150000
```

## 四、主要结果

### 4.1 模型

| 指标 | 数值 |
|---|---|
| 面板 | 9,746,194 行 / 2,622 日 / 5,342 只 / 42 因子 |
| LGBM dev IC | 0.1285（best_iter 85） |
| Stacking dev IC | 0.1316（Ridge 元模型） |
| **5 折 Purged CV** | **0.1041 ± 0.0094，五折全正** |

### 4.2 唯一可落地形态：禁买清单

剔除打分垫底 10%，5 日调仓，单边 15bp：

| 区间 | 年化超额 |
|---|---|
| 2024（meta） | +4.99% |
| 2025H1（dev） | +3.83% |
| 2025-07~2026-10（封存段，一次性） | +2.87% |

### 4.3 被证伪的八条路径

SVM 日频 / SVM 截面 / 多空组合（封存段 −1.08%）/ 纯多头 top10% / 行业动量追涨（0 成本 −5.88%）
/ 基本面因子加成（5 折 4 折变差）/ Muon 优化器（与 AdamW 持平）/ 策略轮动（12 个变体全部跑输固定配置）

### 4.4 关联规则（30 条规则五族）

全部指向「短期过热 + 波动放大」：偏离均线 + 高波动（lift 2.49）、振幅扩张 + 日内强势（2.42）、
近期涨停 + 高于成本（2.40）。**`illiq` 在 184 条规则里只出现 9 次，但 ICIR 权重占 67%**——
线性加权要单调排序，关联规则要极端子集，两者互补。

### 4.5 事件研究

| 事件 | 5 日 CAR | t |
|---|---|---|
| 财报 SUE 极端 | **−1.49%** | **−2.62** |
| 业绩预告（强制披露） | −0.58% | −0.83 |

**越公开、越被提前消化的事件，越没有超额。**

## 五、数据说明（不入库）

仓库只放代码和小样本，完整数据需自行获取：

| 数据 | 规模 | 获取方式 |
|---|---|---|
| 后复权日线 | 5,446 只 / 533 MB | `python scripts/big_download_hfq.py`（akshare） |
| 财报 54 期 × 4 表 | 115 MB | `python scripts/big_fin_download.py` |
| 业绩预告 | 91,126 条 / 54 MB | `python scripts/fetch_yjyg.py` |
| 申万行业 | 31 / 128 类 | `python scripts/swap_industry.py` |
| ST 日历 | 151,170 行 | `python scripts/st_download.py` |

**关键口径**：复权一律用**后复权（hfq）**。东财 A 股前复权是「减法复权」，
茅台历史价会被调成 −143.50，负价区间收益符号是反的。复权断层用**涨跌停制度反查**。

## 六、工程规范

- **测试**：`tests/` 31 个用例，每个对应一个真实 bug（`tests/README.md` 有清单）
- **日志**：所有脚本运行写入 `logs/`，含参数、硬件、耗时、状态（`logs/README.md`）
- **数据字典**：`reports/data_dictionary.csv`，110 个字段
- **可复现性**：换机器重跑，IC 差 0.0005

## 七、团队分工

见 `reports/group_division.md`。

## 八、限制

1. **封存段已消耗**（一次性验证），新模型无法再验证
2. 行业中性化用聚类 31 组而非申万 128 类（申万太细，日均 26 只/行业，噪声盖过信号）
3. 减持公告抓不了（接口限流），改用业绩预告
4. 只有 38 个季度的长线策略，t 值约 2，勉强显著

## 九、TODO

> 完整资产盘点见 `reports/ASSETS_TODO.md`；这里是需要继续做的事。

### 9.1 课程作业（有 deadline，优先）

- [ ] **A1 分类模型**（公告情感 / Jev-like）—— 唯一还没开工的必做项。
      现状：Laya 零样本已验证（`scripts/test_laya.py`，5 条中文公告 4 条判对，22~24 ms/条）；
      4060 全参微调 OOM（门槛约 12 GB），只能 `--freeze-encoder`，要全参得租 4090/A800。
      卡点在标注路径：LLM 弱标注 + 500 条人工抽检（抽检只能由本人做）。
- [x] **A2 关联规则** —— `scripts/assoc_rules.py`：30 条规则五族 + Apriori/FP-Growth 对比 + 9 组敏感性
- [ ] **A3 聚类三件套**（肘部法 / 轮廓系数 / bootstrap ARI）—— 目前只有 `ind_rolling.py`，缺方法论三件套
- [ ] **A4 SHAP / LIME** —— 全局重要性 + 3 个样本的局部解释，模型已有，纯解释层
- [ ] **A5 Notebook** —— 把「下载 → 特征 → 训练 → CV → 清单」串成一键重跑
- [ ] **A6 工程目录重组** —— `tests/`(31) + `logs/` + `requirements.txt` 已就位，缺 `data/ src/ notebooks/ results/` 的标准结构与文档同步
- [ ] **A7 团队分工表** —— `reports/group_division.md` 模板已写两套方案，姓名与贡献 % 待填（只能本人填）
- [ ] **A8 生成 .docx** —— 套六章模板，正文可直接取 `reports/` 现有结论

### 9.2 量化研究（长期，无 deadline）

- [ ] **B1 A2 用 v2 面板重跑** —— 现报的 6.38% / 夏普 0.25 是 hfq-v1 面板数字，断层感知后未重跑
- [ ] **B2 更新 3 份过期报告** —— `abs_return_fix_report.md` / `cv_ridge_report.md` / `long_strategy_report.md` 里的 A2 与基准数字已作废（全市场等权基准 +9.31% → +6.37%）
- [ ] **B3 分析师下调事件** —— `stock_research_report_em` 覆盖不全，先探覆盖率再决定是否做
- [ ] **B4 分钟级 / tick 数据** —— 日频量价已到尽头，下一个 bet 在订单流
- [x] **B5 申万 128 类接入中性化** —— 已评估，结论是不换（日均 26 只/行业，噪声盖过信号）

### 9.3 实盘执行（1 小时/月）

- [x] **C1 A1 月度信号** / **C2 禁买清单** —— `live_screen.py` 已产出当期清单
- [ ] **C3 前瞻跟踪台账** —— 日期 / 目标权重 / 实际成交 / 净值，从 2026-11 开始记
- [ ] **C4 `evaluate_live.py` 对接台账** —— 脚本有，缺数据

### 9.4 清理

- [ ] **D1 删 `data/big/arr_backup.npz`**（2.4 GB，qfq 旧面板，已确认作废）
- [ ] **D2 决定 `arr_hfq_v1.npz` 去留**（2.4 GB，仅作三版对比留档）

### 9.5 永久缺口（不补）

封存段（2025-07~2026-10）已一次性消耗，新模型无法再验证——这是设计取舍，不是待办。
