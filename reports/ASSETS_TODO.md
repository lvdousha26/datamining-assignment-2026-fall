# 资产与待办清单

> 2026-11-09 17:50 盘点｜以磁盘实际状态为准

---

# 第一部分 · 现有资产

## 一、数据资产（5.2 GB 主库 + 532 MB 原始行情）

| 资产 | 规模 | 内容 | 状态 |
|---|---|---|---|
| `data/big/arr.npz` | 2.49 GB | X(2622×5342×42) / Y2 / M / IND / dates / symbols | ✅ 最新（hfq + 断层感知） |
| `data/big/panel_big.parquet` | 1.83 GB | 训练段扁平表，9 个标签列 + 42 因子 | ✅ 与 arr 同源 |
| `data/big/fin_panel.parquet` | 270 MB | 1,109 万行 × 21 财务因子，日频 PIT 展开 | ✅ |
| `data/big/daily_ret.npz` | 22 MB | R(2622×5342) 日收益矩阵 + symbols | ✅ 已清洗 |
| `data/big/lockbox_labels.npz` | 26 MB | 封存段标签（物理隔离） | ✅ 不可重跑 |
| `data/big/raw_hfq/` | 533 MB | 5,446 只后复权日线 | ✅ 原始底稿 |
| `data/big/fin/` | 115 MB | 54 期 × 4 表（资产负债/利润/现金流/业绩） | ✅ |
| `data/big/yjyg/` | 54 MB | 44 期业绩预告，91,126 条 / 5,591 只 | ✅ 新 |
| `data/big/sw_l1.parquet` | 32 KB | 申万一级 31 类 / 5,221 只 | ✅ |
| `data/big/sw_industry.parquet` | 200 KB | 申万二级 128 类 / 5.9 万行 | ✅ |
| `data/big/st_calendar.parquet` | 100 KB | ST 时变日历（151,170 行 / 322 只精确 + 代理） | ✅ |
| `data/big/idx_daily.pkl` | 139 KB | 沪深300 + 中证全指日线 | ✅ |
| `data/us/` | 949 KB | 10 只美股后复权 + 纳指 | ✅ |

**冗余**：`arr_backup.npz`（2.4 GB，qfq 旧面板）+ `arr_hfq_v1.npz`（2.4 GB，断层感知前）可清理，建议只留后者。

## 二、模型资产

| 资产 | 内容 | 指标 |
|---|---|---|
| `checkpoints/lgbm_big.txt` | LightGBM，best_iter=85 | dev IC **0.1285** |
| `checkpoints/tf_seed{1,2,3}.pt` | Transformer 3 seed（seq40/d96/4头/2层/SwiGLU/RMSNorm） | 单模型 dev IC 约 0.13 |
| `reports/big_models_meta.json` | Ridge 元模型系数 + 标准化参数 + 因子权重 | Stacking dev IC **0.1316** |
| 5 折 Purged CV | `reports/purged_cv_folds.csv` | **0.1041 ± 0.0094**，5 折全正 |

## 三、策略资产

| 策略 | 业绩 | 状态 |
|---|---|---|
| **A1 策略③**（六腿 ETF） | 11.92% / 9.40% / **夏普 1.06** / −10.3%（2016-2026，130 月） | ✅ 可实盘 |
| **A2 质量+估值** | 6.38% / 17.76% / 夏普 0.25 / −37.5%（**hfq-v1 面板，待用 v2 重跑**） | ⚠️ 数字待更新 |
| **禁买清单**（剔底 10%） | dev 2025H1 **+4.67%**、meta 2024 **+5.85%** | ✅ 可用 |
| 行业内剔底（聚类 31 组） | +5.57% / 夏普 5.16（2024-2025H1 口径） | ✅ 备选 |
| `live/screen_2026-10-30.csv` | 5,000+ 只逐股打分，D1 500 只 | ✅ 当期清单 |

## 四、代码资产（82 个脚本）

| 类别 | 数量 | 核心 |
|---|---|---|
| 数据管线 | 12 | `big_download_hfq` / `big_features` / `big_fin_download` / `big_fin_features` / `fetch_yjyg` / `st_*` / `swap_industry` |
| 模型训练 | 10 | `big_train` / `tf_net` / `stack_ridge` / `muon_*` / `purged_cv` |
| 验证评估 | 9 | `lockbox_eval` / `evaluate_screen` / `ridge_check` / `audit_split` / `compare_rebuild` / `build_data_dict` |
| 事件研究 | 3 | `event_study` / `make_events` / `fetch_yjyg` |
| 策略实现 | 8 | `long_strategy` / `long_strategy_l1` / `strategy_compare` / `A2_daily_sim` / `combine_v2` / `live_screen` |
| 因子审计 | 6 | `factor_scorecard` / `audit_data` / `audit_overfit` / `audit_redundancy` |
| 基础设施 | 5 | `remote` / `deploy_v2` / `train_remote` / `checkpoint` |

BASE 目录另有 92 个（策略③及历史实验），合计 **174 个脚本**。

## 五、报告资产（43 份 md）

按主题：数据修复 3、策略手册 2、事件研究 6、因子评审 5、模型验证 6、策略实验 10、其他 11。

---

# 第二部分 · 待办清单

## A. 课程作业（有 deadline，最紧急）

| # | 任务 | 缺口 | 工作量 | 谁能做 |
|---|---|---|---|---|
| A1 | **分类模型** | ✅ 完成：零样本 60.8% → 冻结微调 78.7%（63 秒），但输给正则 84.6% | — | 已交付 |
| A2 | **关联规则**（FP-Growth + Apriori 对比 + 敏感性） | ✅ **已完成**：30 条规则五族，Apriori 反超 FP-Growth 9.5 倍 | — | 已交付 |
| A3 | **聚类三件套** | ✅ 已完成：k=4 拐点、ARI@31 只有 0.392 | — | 已交付 |
| A4 | **SHAP**（全局 + 3 样本） | ✅ 已完成：TreeSHAP 精确值，size/turn5/pos250 前三 | — | 已交付 |
| A5 | **Notebook**（可一键重跑）或仪表盘 | ❌ 未做 | 1.5h | 我 |
| A6 | **工程目录**（data/src/notebooks/results/docs + README + requirements + 断言 + 运行日志） | ⚠️ 部分：`tests/` 31 用例 + `logs/` 运行日志已就位，缺目录重组与 requirements | 1h | 我 |
| A7 | **团队分工表 + 沟通记录** | ⚠️ 模板已就绪（`reports/group_division.md`），姓名与贡献% 待填 | — | **只能你填** |
| A8 | **生成 .docx**（套模板六章） | ❌ 未做 | 2h | 我 |

**A1 需要你先定三件事**：
1. 单人交还是组队（决定 A7 怎么写）
2. Jev 方案：① LLaMA-Factory + 标签 logit（半天，8G 显存够）还是 ③ ①+真 Jev 对照（3 天）
3. 标注路径：LLM 弱标注 + 500 条人工抽检（只有这 500 条必须你亲自做）

**已达标、无需补**：数据获取（≥2 类异构 / ≥100 万条 ✅ 974 万 / ≥20 字段 ✅ 110）、数据治理（防泄漏做到 lockbox + Purged CV，远超要求）、数据字典 ✅。

## B. 量化研究（长期，无 deadline）

| # | 任务 | 原因 | 工作量 |
|---|---|---|---|
| B1 | **A2 用 v2 面板重跑** | 现在的 6.38% 是 hfq-v1 面板算的，断层感知后未重跑 | 30 min |
| B2 | **更新 3 份过期报告** | `abs_return_fix` / `cv_ridge` / `long_strategy` 里 A2 与基准数字已作废 | 1h |
| B3 | **分析师下调事件** | `stock_research_report_em` 覆盖不全，需先探覆盖率 | 3h |
| B4 | **分钟级 / tick 数据** | 日频量价已到尽头，下一个 bet 在订单流 | 1-2 周 |
| B5 | 把 `sw_industry`（128 类）接进因子中性化 | 已接入 IND，但中性化仍用聚类 31 组（申万太细反而更差） | 已完成评估，结论是不换 |

**永久缺口**：lockbox 已消耗，新模型无法再验（设计如此，不补）。

## C. 实盘执行（月度，1 小时/月）

| # | 任务 | 状态 |
|---|---|---|
| C1 | A1 月度信号 | ✅ `live_signal.py` 已有（BASE 目录） |
| C2 | 禁买清单 | ✅ `live_screen.py` 已有 |
| C3 | **前瞻跟踪台账**（日期/目标权重/实际成交/净值） | ❌ **未建，从 2026-11 开始** |
| C4 | `evaluate_live.py` 对接台账 | ⚠️ 有脚本，无数据 |

## D. 清理（10 分钟）

| # | 任务 |
|---|---|
| D1 | 删 `data/big/arr_backup.npz`（2.4 GB，qfq 旧面板，已确认作废） |
| D2 | 保留 `arr_hfq_v1.npz`（对比用），或也删（省 2.4 GB） |

## E. 已完成（本轮新增）

| 项 | 产物 |
|---|---|
| 自动化测试 | `tests/` 31 个用例（26 快 + 5 慢），10.7 秒全过 |
| 运行日志 | `logs/` + `scripts/runlog.py`，8 个脚本已接入 |
| 数据字典 | `reports/data_dictionary.csv/.md`，110 字段 |
| 事件框架 | `event_study.py` + `make_events.py`，4 类事件 |
| 关联规则 | `assoc_rules.py`，30 条规则五族 + 敏感性 9 组 |
| 分工模板 | `reports/group_division.md`，两套方案 |

---

# 第三部分 · 建议顺序

```
第 1 步（今天）   A7 你去问老师：单人能否交  ← 唯一需要外部确认的事
第 2 步（2h）     A2 关联规则 + A3 聚类三件套 + A4 SHAP   ← 纯跑，可并行
第 3 步（1.5h）   A5 Notebook + A6 工程目录
第 4 步（你+我）  A1 分类模型（先定方案）
第 5 步（2h）     A8 生成 docx
第 6 步（30min）  B1 A2 重跑 + B2 报告更新   ← 数据一致性
第 7 步（10min）  D1/D2 清理
```

## 一句话

**课程作业还差 7 项，其中 6 项我能做、1 项只能你做（分工表）；量化侧只剩「数字过期」和一个「永久缺口」；实盘缺一个台账。**

比资产更值钱的是那 43 份报告里的负结果——SVM、多空、行业动量、基本面加成、Muon、TAA、策略轮动、业绩预告事件，**八条路全部证伪**，这是简历上比任何一条收益曲线都硬的东西。
